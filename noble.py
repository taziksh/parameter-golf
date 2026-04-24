"""
PyTorch port of noble_mlx.py (NOBLE: nonlinear low-rank branch).

Paper (arXiv:2603.06492) variant: for each linear layer xW + b, augment with
  y = xW + b + lora_up(cos_net(lora_down(x)))
where cos_net = second_cos(fc(first_cos(x))) with two learnable CosActivations
around a rank-by-rank mixing matrix.

Init + lr-mult scheme mirrors noble_mlx.py exactly. See that file for design
notes (why Adam for NOBLE branch 2D, why `(d/r)^gamma` lr_mults, etc.).

Weight convention mirrors CastedLinear in train_gpt.py:
  - 2D weights are fp32 nn.Parameters, cast to x.dtype at matmul time.
  - 1D biases / freq / phase are fp32 nn.Parameters, cast at use time.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


FREQ_LR_MULT = 2.0
PHASE_LR_MULT = 4.0


class CosActivation(nn.Module):
    def __init__(
        self,
        dim: int,
        min_freq: float = 0.8,
        max_freq: float = 1.2,
        phase_init_std: float = 0.1,
    ):
        super().__init__()
        freqs = torch.empty(dim).uniform_(min_freq, max_freq)
        phase = torch.randn(dim) * phase_init_std
        phase = torch.fmod(phase, 2.0 * math.pi)
        self.freq_scale = nn.Parameter(freqs.float())
        self.freq_bias = nn.Parameter(phase.float())

    def forward(self, x: Tensor) -> Tensor:
        fs = self.freq_scale.to(x.dtype)
        fb = self.freq_bias.to(x.dtype)
        return torch.cos(fs * x + fb)


class CosNet(nn.Module):
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
        bound = 1.0 / math.sqrt(dim)
        self.fc_weight = nn.Parameter((torch.randn(dim, dim) * std).float())
        self.fc_bias = nn.Parameter(torch.empty(dim).uniform_(-bound, bound).float())

        self._fc_lr_mult = float((full_dim / dim) ** lr_mult_power)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(x)
        h = F.linear(h, self.fc_weight.to(x.dtype), self.fc_bias.to(x.dtype))
        return self.second_act(h)


class NOBLELinear(nn.Module):
    """
    Paper-faithful per-linear NOBLE: y = xW + b + lora_up(cos_net(lora_down(x))).

    Drop-in replacement for CastedLinear. `self.weight` shape is (out, in) to
    match nn.Linear. Bias defaults to False to match CastedLinear use in this
    codebase. The main linear uses the NOBLE paper init (linear_init_scale /
    sqrt(in)), NOT nn.Linear's default, so callers should not also apply the
    baseline's `_zero_init` override.
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
        self.weight = nn.Parameter((torch.randn(out_features, in_features) * std_lin).float())
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

        std_down = 1.0 / math.sqrt(in_features)
        self.lora_down_weight = nn.Parameter((torch.randn(lora_rank, in_features) * std_down).float())
        self.lora_down_bias = nn.Parameter(torch.zeros(lora_rank, dtype=torch.float32))

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
        self.lora_up_weight = nn.Parameter((torch.randn(out_features, lora_rank) * std_up).float())

        self._lora_up_lr_mult = float((full_dim / lora_rank) ** (lora_lr_mult_power * 2.0))

    def forward(self, x: Tensor) -> Tensor:
        main = F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.use_bias else None)
        h = F.linear(x, self.lora_down_weight.to(x.dtype), self.lora_down_bias.to(x.dtype))
        h = self.cos_net(h)
        h = F.linear(h, self.lora_up_weight.to(x.dtype))
        return main + h

    def param_lr_mults(self, prefix: str) -> dict[str, float]:
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


def is_noble_branch_key(name: str) -> bool:
    """True for any 2D or 1D param belonging to the NOBLE branch (lora_* or cos_net.*)."""
    return "lora_" in name or ".cos_net." in name


def noble_lr_mults_for_gpt_linears(
    model: nn.Module,
    attn_names: tuple[str, ...] = ("c_q", "c_k", "c_v", "proj"),
    mlp_names: tuple[str, ...] = ("fc", "proj"),
) -> dict[str, float]:
    """
    Collect per-param lr_mults keyed by the dotted paths produced by
    `model.named_parameters()`. Only populated where a NOBLELinear actually
    sits; returns {} if the model has no NOBLE layers.
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
