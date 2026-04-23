"""
MLX port of NOBLE (https://github.com/ethansmith2000/noble).

Paper (arXiv:2603.06492) variant: for each linear layer xW + b, augment with
  y = xW + b + lora_up(cos_net(lora_down(x)))
where cos_net = second_cos(fc(first_cos(·))) with two distinct learnable
cosine activations around a rank-by-rank mixing matrix. `lora_up` and
cos_net's `fc` get learning-rate multipliers that scale with `(dim / rank)`.

This module exposes two attachment styles:
  - `NOBLELinear`: the paper-faithful per-linear variant (§4.1.1 of the
    paper: "NOBLE is applied to all linear projections").
  - `NOBLEBranch`: the branch alone, intended for per-block / whole-model
    attachment per Ethan's Twitter claim (2026-03-23) that per-block and
    whole-model work about equally well. Unverified outside that claim.

Prefer `NOBLELinear` for reproducing the paper.

Weight conventions follow train_gpt_mlx.py's `CastedLinear`:
  - 2D weights stored as fp32 mx.array attributes shaped (out, in).
  - 1D biases / freq / phase stored as fp32 mx.array attributes.
  - Cast to x.dtype on forward so compute runs in COMPUTE_DTYPE (bf16).

Integration sketch (per-linear, paper-faithful):

    from noble_mlx import NOBLELinear, noble_lr_mults_for_gpt_linears

    # 1. In CausalSelfAttention / MLP, swap CastedLinear(in, out) for
    #    NOBLELinear(in, out, lora_rank=R) when use_noble is on.
    # 2. In SplitOptimizers:
    #    (a) compute lr_mults once via noble_lr_mults_for_gpt_linears(model);
    #    (b) EXCLUDE every NOBLE branch 2D weight ("lora_" or ".cos_net.") from
    #        matrix_keys so Muon never touches them;
    #    (c) bucket the NOBLE 2D branch weights PLUS the 1D scalars by distinct
    #        lr_mult and run one optim.Adam per bucket.
    #
    # Why route NOBLE's 2D weights through Adam? The paper uses AdamW throughout;
    # its lr_mults (d/r)^gamma were derived for Adam's v-normalized update
    # magnitude. Muon's update is already magnitude-normalized (orthogonalized
    # + sqrt(shape[0]/shape[1]) scaling), so stacking lr_mult > 1 on top of
    # Muon's matrix_lr blows the weights up at step 1-2 (observed: step 2 loss
    # spike to 17+ from 7, no recovery). Routing NOBLE's branch 2D through
    # Adam is what matches the paper.
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


# Per-param lr multipliers from upstream CosActivation (noble_simple.py).
FREQ_LR_MULT = 2.0
PHASE_LR_MULT = 4.0


def _fmod(x: mx.array, y: float) -> mx.array:
    """
    Truncation-style modulo (same semantics as torch.fmod / C fmod): the
    result has the same sign as `x`. MLX's native `%` is floor-modulo, which
    differs for negative inputs.
    """
    return x - mx.sign(x) * mx.floor(mx.abs(x) / y) * y


class CosActivation(nn.Module):
    """
    cos(freq_scale * x + freq_bias) with per-dim learnable frequency and phase.
    Initialization exactly matches noble_simple.py:
      freq  ~ Uniform(min_freq, max_freq)
      phase ~ Normal(0, phase_init_std), then fmod by 2*pi.
    """

    def __init__(
        self,
        dim: int,
        min_freq: float = 0.8,
        max_freq: float = 1.2,
        phase_init_std: float = 0.1,
    ):
        super().__init__()
        freqs = mx.random.uniform(low=min_freq, high=max_freq, shape=(dim,))
        phase = mx.random.normal(shape=(dim,)) * phase_init_std
        phase = _fmod(phase, 2.0 * math.pi)
        self.freq_scale = freqs.astype(mx.float32)
        self.freq_bias = phase.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        fs = self.freq_scale.astype(x.dtype)
        fb = self.freq_bias.astype(x.dtype)
        return mx.cos(fs * x + fb)


class CosNet(nn.Module):
    """
    Paper `Net` wrapper: second_act(fc(first_act(x))) with two distinct
    CosActivations and an inner Linear(rank, rank). Matches noble_simple.py:
      fc.weight ~ Normal(0, init_scale / sqrt(rank))
      fc.bias   ~ Uniform(-1/sqrt(rank), 1/sqrt(rank))  (PyTorch default)
      fc.weight learning rate is multiplied by (full_dim / rank)^lr_mult_power.

    Upstream attaches the fc lr_mult to the module, not the weight; that looks
    like a bug since their optimizer helper reads `getattr(param, 'lr_mult')`.
    This port attaches it to the weight (the intended semantic).
    """

    def __init__(
        self,
        dim: int,
        full_dim: int,
        init_scale: float = 0.5,
        lr_mult_power: float = 0.5,
        min_freq: float = 0.8,
        max_freq: float = 1.2,
        phase_init_std: float = 0.1,
    ):
        super().__init__()
        self.act = CosActivation(dim, min_freq=min_freq, max_freq=max_freq, phase_init_std=phase_init_std)
        self.second_act = CosActivation(dim, min_freq=min_freq, max_freq=max_freq, phase_init_std=phase_init_std)

        std = init_scale / math.sqrt(dim)
        self.fc_weight = (mx.random.normal(shape=(dim, dim)) * std).astype(mx.float32)
        bound = 1.0 / math.sqrt(dim)
        self.fc_bias = mx.random.uniform(low=-bound, high=bound, shape=(dim,)).astype(mx.float32)

        self._fc_lr_mult = float((full_dim / dim) ** lr_mult_power)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.act(x)
        h = h @ self.fc_weight.astype(x.dtype).T + self.fc_bias.astype(x.dtype)
        return self.second_act(h)


class NOBLEBranch(nn.Module):
    """
    Nonlinear low-rank branch, branch-only: `lora_up(cos_net(lora_down(x)))`.
    Designed for per-block or whole-model attachment (`y_base + branch(x)`).

    Initialization exactly matches noble_simple.py's NOBLELinearCosNet:
      lora_down.weight ~ Normal(0, 1/sqrt(dim))
      lora_down.bias   = zeros
      lora_up.weight   ~ Normal(0, lora_up_init_scale/sqrt(rank))
      lora_up has no bias.

    Onesided learning-rate multiplier (paper variant, always on):
      lora_up.weight gets lr_mult = (dim / rank)^(lora_lr_mult_power * 2).
      lora_down.weight stays at 1.0.
    """

    def __init__(
        self,
        dim: int,
        lora_rank: int = 32,
        lora_lr_mult_power: float = 0.2,
        lora_up_init_scale: float = 0.01,
        cos_net_init_scale: float = 0.5,
        cos_net_lr_mult_power: float = 0.5,
        cos_min_freq: float = 0.8,
        cos_max_freq: float = 1.2,
        cos_phase_init_std: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.lora_rank = lora_rank

        std_down = 1.0 / math.sqrt(dim)
        self.lora_down_weight = (mx.random.normal(shape=(lora_rank, dim)) * std_down).astype(mx.float32)
        self.lora_down_bias = mx.zeros((lora_rank,), dtype=mx.float32)

        self.cos_net = CosNet(
            dim=lora_rank,
            full_dim=dim,
            init_scale=cos_net_init_scale,
            lr_mult_power=cos_net_lr_mult_power,
            min_freq=cos_min_freq,
            max_freq=cos_max_freq,
            phase_init_std=cos_phase_init_std,
        )

        std_up = lora_up_init_scale / math.sqrt(lora_rank)
        self.lora_up_weight = (mx.random.normal(shape=(dim, lora_rank)) * std_up).astype(mx.float32)

        self._lora_up_lr_mult = float((dim / lora_rank) ** (lora_lr_mult_power * 2.0))

    def __call__(self, x: mx.array) -> mx.array:
        h = x @ self.lora_down_weight.astype(x.dtype).T + self.lora_down_bias.astype(x.dtype)
        h = self.cos_net(h)
        return h @ self.lora_up_weight.astype(x.dtype).T

    def param_lr_mults(self, prefix: str) -> dict[str, float]:
        """
        Return {full_param_name: lr_mult} for every param under this branch
        whose lr_mult differs from 1.0. Keys use tree_flatten dot-paths.
        """
        def k(*parts: str) -> str:
            return ".".join((prefix, *parts))
        return {
            k("lora_up_weight"): self._lora_up_lr_mult,
            k("cos_net", "fc_weight"): self.cos_net._fc_lr_mult,
            k("cos_net", "act", "freq_scale"): FREQ_LR_MULT,
            k("cos_net", "act", "freq_bias"): PHASE_LR_MULT,
            k("cos_net", "second_act", "freq_scale"): FREQ_LR_MULT,
            k("cos_net", "second_act", "freq_bias"): PHASE_LR_MULT,
        }


class NOBLELinear(nn.Module):
    """
    Paper-faithful per-linear NOBLE:  y = xW + b + lora_up(cos_net(lora_down(x))).

    Drop-in (shape-compatible) replacement for CastedLinear. The main linear
    weight is stored as `self.weight` (shape (out, in), fp32) so existing
    code like `b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)` keeps
    working. Main-linear bias defaults to False to match CastedLinear.

    Initialization exactly matches noble_simple.py's NOBLELinearCosNet:
      linear.weight   ~ Normal(0, linear_init_scale/sqrt(in))
      lora_down.weight ~ Normal(0, 1/sqrt(in))
      lora_down.bias   = zeros
      lora_up.weight   ~ Normal(0, lora_up_init_scale/sqrt(rank))
      lora_up has no bias.

    Onesided learning-rate multiplier (paper variant, always on):
      lora_up.weight gets lr_mult = (min(in,out) / rank)^(lora_lr_mult_power * 2).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int = 32,
        lora_lr_mult_power: float = 0.2,
        lora_up_init_scale: float = 0.01,
        linear_init_scale: float = 0.5,
        cos_net_init_scale: float = 0.5,
        cos_net_lr_mult_power: float = 0.5,
        cos_min_freq: float = 0.8,
        cos_max_freq: float = 1.2,
        cos_phase_init_std: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.use_bias = bias
        full_dim = min(in_features, out_features)

        std_lin = linear_init_scale / math.sqrt(in_features)
        self.weight = (mx.random.normal(shape=(out_features, in_features)) * std_lin).astype(mx.float32)
        if bias:
            self.bias = mx.zeros((out_features,), dtype=mx.float32)

        std_down = 1.0 / math.sqrt(in_features)
        self.lora_down_weight = (mx.random.normal(shape=(lora_rank, in_features)) * std_down).astype(mx.float32)
        self.lora_down_bias = mx.zeros((lora_rank,), dtype=mx.float32)

        self.cos_net = CosNet(
            dim=lora_rank,
            full_dim=full_dim,
            init_scale=cos_net_init_scale,
            lr_mult_power=cos_net_lr_mult_power,
            min_freq=cos_min_freq,
            max_freq=cos_max_freq,
            phase_init_std=cos_phase_init_std,
        )

        std_up = lora_up_init_scale / math.sqrt(lora_rank)
        self.lora_up_weight = (mx.random.normal(shape=(out_features, lora_rank)) * std_up).astype(mx.float32)

        self._lora_up_lr_mult = float((full_dim / lora_rank) ** (lora_lr_mult_power * 2.0))

    def __call__(self, x: mx.array) -> mx.array:
        main = x @ self.weight.astype(x.dtype).T
        if self.use_bias:
            main = main + self.bias.astype(x.dtype)
        h = x @ self.lora_down_weight.astype(x.dtype).T + self.lora_down_bias.astype(x.dtype)
        h = self.cos_net(h)
        h = h @ self.lora_up_weight.astype(x.dtype).T
        return main + h

    def param_lr_mults(self, prefix: str) -> dict[str, float]:
        """
        Return {full_param_name: lr_mult} for every param under this linear
        whose lr_mult differs from 1.0. Keys use tree_flatten dot-paths.
        """
        def k(*parts: str) -> str:
            return ".".join((prefix, *parts))
        return {
            k("lora_up_weight"): self._lora_up_lr_mult,
            k("cos_net", "fc_weight"): self.cos_net._fc_lr_mult,
            k("cos_net", "act", "freq_scale"): FREQ_LR_MULT,
            k("cos_net", "act", "freq_bias"): PHASE_LR_MULT,
            k("cos_net", "second_act", "freq_scale"): FREQ_LR_MULT,
            k("cos_net", "second_act", "freq_bias"): PHASE_LR_MULT,
        }


def noble_lr_mults_for_gpt_linears(
    model: nn.Module,
    attn_names: tuple[str, ...] = ("c_q", "c_k", "c_v", "proj"),
    mlp_names: tuple[str, ...] = ("fc", "proj"),
) -> dict[str, float]:
    """
    Collect per-param lr_mults for a GPT whose blocks' attention and MLP linear
    slots are (optionally) NOBLELinear instances. Returns a dict keyed by the
    dotted paths tree_flatten(model.parameters()) produces.
    """
    out: dict[str, float] = {}
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return out
    for i, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        if attn is not None:
            for name in attn_names:
                m = getattr(attn, name, None)
                if isinstance(m, NOBLELinear):
                    out.update(m.param_lr_mults(f"blocks.{i}.attn.{name}"))
        mlp = getattr(block, "mlp", None)
        if mlp is not None:
            for name in mlp_names:
                m = getattr(mlp, name, None)
                if isinstance(m, NOBLELinear):
                    out.update(m.param_lr_mults(f"blocks.{i}.mlp.{name}"))
    return out


def is_noble_branch_key(name: str) -> bool:
    """
    True if `name` is a 2D NOBLE branch weight that must be excluded from Muon
    routing (lora_down_weight, lora_up_weight, cos_net.fc_weight). The CosNet
    internal linear is nested under `.cos_net.`, the lora matrices contain
    `lora_` — together those substrings identify every 2D branch weight.
    """
    return "lora_" in name or ".cos_net." in name


def noble_lr_mults_for_gpt_blocks(
    model: nn.Module,
    attr_name: str = "noble_branch",
) -> dict[str, float]:
    """
    Collect per-param lr_mults for a GPT whose Block instances each carry a
    NOBLEBranch at `block.{attr_name}`. Returns a dict keyed by the dotted
    paths that `tree_flatten(model.parameters())` produces, so it can be
    indexed directly inside the optimizer step.
    """
    out: dict[str, float] = {}
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return out
    for i, block in enumerate(blocks):
        branch = getattr(block, attr_name, None)
        if isinstance(branch, NOBLEBranch):
            out.update(branch.param_lr_mults(f"blocks.{i}.{attr_name}"))
    return out


# -----------------------------------------------------------------------------
# Optimizer plumbing guidance (paper-faithful: Adam, not Muon, for branch 2D).
# -----------------------------------------------------------------------------
# 1. From matrix_keys (Muon routing), EXCLUDE everything where
#    `is_noble_branch_key(k)` is True. That sends the main linear's `.weight`
#    to Muon (as before) and keeps lora_down / lora_up / cos_net.fc OUT of
#    Muon. Reasons: the paper uses AdamW for these, and Muon's
#    magnitude-normalized updates blow up when stacked with lr_mult > 1.
#
# 2. Combine the remaining 2D NOBLE branch weights WITH the existing 1D
#    scalar keys, then bucket the union by distinct lr_mult value and run
#    one optim.Adam per bucket (pre-scaling grads does not work because Adam
#    normalizes by sqrt(v)).
#
#     buckets: dict[float, list[str]] = {}
#     for k in (noble_2d_keys + scalar_keys):
#         buckets.setdefault(self.lr_mults.get(k, 1.0), []).append(k)
#     self.adam_buckets = [
#         (mult, keys, optim.Adam(learning_rate=args.scalar_lr, betas=[...], eps=..., bias_correction=True))
#         for mult, keys in buckets.items()
#     ]
#
# In step():
#
#     for mult, keys, adam in self.adam_buckets:
#         adam.learning_rate = self.args.scalar_lr * lr_mul * mult
#         updated.update(adam.apply_gradients(
#             {k: grads[k] for k in keys},
#             {k: params[k] for k in keys},
#         ))
#
# For per-linear NOBLE at defaults, you'll see ~5 buckets:
#   1.0  (everything else, plus lora_down_weight, lora_down_bias, cos_net.fc_bias)
#   2.0  (freq_scale)
#   4.0  (freq_bias)
#   (min/r)^0.4  (lora_up_weight)
#   (min/r)^0.5  (cos_net.fc_weight)
# For the 512->256 shapes (c_k, c_v), the last two buckets use min=256 instead
# of 512, so those end up as separate buckets from the 512->512 / 512->1024
# ones — expect closer to 7-8 buckets total across a mixed-shape model.
