"""Microbenchmark v2: bigbag attn (FA3 dense) vs intra-doc variants.

v2 addresses four weaknesses in v1:
  1. More iterations and warmup so the per-iter measurement is statistically
     stable (timer noise + GPU clock drift dominate at small N).
  2. Re-randomized cu_seqlens / position_ids each iter, mirroring real
     training (every batch has a different doc layout).
  3. Forward + backward, since training does both. (Forward-only inflates
     the relative cost of attention vs the rest of the step.)
  4. Mean +/- std and a sanity print of per-iter timings, so we can eyeball
     whether the delta is real or noise.

Run: python3 bench_attn.py
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn_interface import flash_attn_func, flash_attn_varlen_func


# bigbag training config (1xH100, per-GPU micro-batch).
B = 48
T = 2048
H = 8
H_KV = 4
HEAD_DIM = 64
DIM = H * HEAD_DIM
ROPE_DIMS = 16
ROPE_BASE = 1e4
QK_GAIN = 5.25
DTYPE = torch.bfloat16
DEVICE = 'cuda'
N_LAYERS = 11

MAX_SUBDOCS = 16

# Bench knobs:
#   WARMUP: enough iters for torch.compile to settle (compilation +
#     Inductor autotune both run during the first several real calls).
#   ITERS:  sized so total bench wall time per variant is several seconds,
#     making sub-percent deltas resolvable above timer / clock-drift noise.
WARMUP = 20
ITERS = 200


def apply_rope_partial(x, cos, sin, rope_dims):
    x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
    half = rope_dims // 2
    x1, x2 = x_rope[..., :half], x_rope[..., half:]
    x_rot = torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)
    return torch.cat([x_rot, x_pass], dim=-1)


# Variant 1: baseline (FA3 dense + cached cos/sin)
class CachedRotary(nn.Module):
    def __init__(self, head_dim, base, train_seq_len, rope_dims):
        super().__init__()
        self.rope_dims = rope_dims
        inv_freq = 1.0 / base ** (
            torch.arange(0, rope_dims, 2, dtype=torch.float32) / rope_dims
        )
        t = torch.arange(train_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos', freqs.cos()[None, :, None, :], persistent=False)
        self.register_buffer('sin', freqs.sin()[None, :, None, :], persistent=False)

    def forward(self, seq_len, dtype):
        return self.cos[:, :seq_len].to(dtype), self.sin[:, :seq_len].to(dtype)


class BaselineAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_q = nn.Linear(DIM, DIM, bias=False)
        self.c_k = nn.Linear(DIM, H_KV * HEAD_DIM, bias=False)
        self.c_v = nn.Linear(DIM, H_KV * HEAD_DIM, bias=False)
        self.proj = nn.Linear(DIM, DIM, bias=False)
        self.q_gain = nn.Parameter(torch.full((H,), QK_GAIN))
        self.rotary = CachedRotary(HEAD_DIM, ROPE_BASE, T, ROPE_DIMS)

    def forward(self, x):
        B_, T_, _ = x.shape
        q = self.c_q(x).reshape(B_, T_, H, HEAD_DIM)
        k = self.c_k(x).reshape(B_, T_, H_KV, HEAD_DIM)
        v = self.c_v(x).reshape(B_, T_, H_KV, HEAD_DIM)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T_, q.dtype)
        q = apply_rope_partial(q, cos, sin, ROPE_DIMS)
        k = apply_rope_partial(k, cos, sin, ROPE_DIMS)
        q = q * self.q_gain.to(q.dtype)[None, None, :, None]
        y = flash_attn_func(q, k, v, causal=True)
        return self.proj(y.reshape(B_, T_, DIM))


# Variant 2: intra-doc + LUT gather RoPE
class LutRotary(nn.Module):
    def __init__(self, head_dim, base, max_pos, rope_dims):
        super().__init__()
        self.rope_dims = rope_dims
        inv_freq = 1.0 / base ** (
            torch.arange(0, rope_dims, 2, dtype=torch.float32) / rope_dims
        )
        positions = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer('cos_lut', freqs.cos(), persistent=False)
        self.register_buffer('sin_lut', freqs.sin(), persistent=False)

    def forward(self, position_ids, dtype):
        cos = self.cos_lut[position_ids].unsqueeze(2).to(dtype)
        sin = self.sin_lut[position_ids].unsqueeze(2).to(dtype)
        return cos, sin


class IntradocLutAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_q = nn.Linear(DIM, DIM, bias=False)
        self.c_k = nn.Linear(DIM, H_KV * HEAD_DIM, bias=False)
        self.c_v = nn.Linear(DIM, H_KV * HEAD_DIM, bias=False)
        self.proj = nn.Linear(DIM, DIM, bias=False)
        self.q_gain = nn.Parameter(torch.full((H,), QK_GAIN))
        self.rotary = LutRotary(HEAD_DIM, ROPE_BASE, T, ROPE_DIMS)

    def forward(self, x, cu_seqlens, position_ids):
        B_, T_, _ = x.shape
        q = self.c_q(x).reshape(B_, T_, H, HEAD_DIM)
        k = self.c_k(x).reshape(B_, T_, H_KV, HEAD_DIM)
        v = self.c_v(x).reshape(B_, T_, H_KV, HEAD_DIM)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(position_ids, q.dtype)
        q = apply_rope_partial(q, cos, sin, ROPE_DIMS)
        k = apply_rope_partial(k, cos, sin, ROPE_DIMS)
        q = q * self.q_gain.to(q.dtype)[None, None, :, None]
        q_flat = q.reshape(B_ * T_, H, HEAD_DIM)
        k_flat = k.reshape(B_ * T_, H_KV, HEAD_DIM)
        v_flat = v.reshape(B_ * T_, H_KV, HEAD_DIM)
        batch_offsets = torch.arange(B_, device=x.device, dtype=torch.int32) * T_
        cu_flat = (cu_seqlens + batch_offsets[:, None]).reshape(-1)
        y = flash_attn_varlen_func(
            q_flat, k_flat, v_flat,
            cu_seqlens_q=cu_flat, cu_seqlens_k=cu_flat,
            max_seqlen_q=T_, max_seqlen_k=T_, causal=True,
        )
        return self.proj(y.reshape(B_, T_, DIM))


# Variant 3: intra-doc + trig-on-the-fly RoPE (kept for comparison)
class TrigRotary(nn.Module):
    def __init__(self, head_dim, base, rope_dims):
        super().__init__()
        self.rope_dims = rope_dims
        inv_freq = 1.0 / base ** (
            torch.arange(0, rope_dims, 2, dtype=torch.float32) / rope_dims
        )
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, position_ids, dtype):
        freqs = position_ids[..., None].float() * self.inv_freq[None, None, :]
        cos = freqs.cos().unsqueeze(2).to(dtype)
        sin = freqs.sin().unsqueeze(2).to(dtype)
        return cos, sin


class IntradocTrigAttn(IntradocLutAttn):
    def __init__(self):
        super().__init__()
        self.rotary = TrigRotary(HEAD_DIM, ROPE_BASE, ROPE_DIMS)


def make_intradoc_meta(rng):
    """One fresh (cu, pos) pair, mimicking the real per-window distribution.

    Real measurement (sp8192 train shard 0): mean 2.5 BOS per 2048-window,
    p99 = 7, max = 11. We sample n ~ Poisson(2.5) and truncate to fit.
    """
    cu = np.full((B, MAX_SUBDOCS + 1), T, dtype=np.int32)
    pos = np.empty((B, T), dtype=np.int32)
    for b in range(B):
        n = max(1, int(rng.poisson(2.5)))
        n = min(n, MAX_SUBDOCS - 1)  # cap at admittable count
        bps = sorted(rng.integers(1, T, size=n - 1)) if n > 1 else []
        cu[b, 0] = 0
        for i, p in enumerate(bps):
            cu[b, 1 + i] = p
        cu[b, 1 + len(bps)] = T
        starts = [0] + list(bps) + [T]
        for i in range(len(starts) - 1):
            seg_len = starts[i + 1] - starts[i]
            pos[b, starts[i]:starts[i + 1]] = np.arange(seg_len, dtype=np.int32)
    return torch.from_numpy(cu).cuda(), torch.from_numpy(pos).cuda()


def bench_fwd_bwd(name, model, fixed_input, intradoc, n_layers=N_LAYERS):
    """Time forward + backward for one variant, with fresh metadata each iter
    if intradoc=True. Returns (mean_ms, std_ms, ms_samples)."""

    def stacked(x, *args):
        for _ in range(n_layers):
            x = model(x, *args)
        return x

    compiled = torch.compile(stacked, fullgraph=True, dynamic=False)

    rng = np.random.default_rng(0)

    def step():
        x = fixed_input.detach().requires_grad_(True)
        if intradoc:
            cu, pos = make_intradoc_meta(rng)
            out = compiled(x, cu, pos)
        else:
            out = compiled(x)
        # Use sum() rather than a real loss; we just want a backward pass.
        out.float().sum().backward()

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()

    samples = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples = np.array(samples)
    mean = samples.mean()
    std = samples.std(ddof=1)
    sem = std / np.sqrt(len(samples))
    print(f"  {name:32s}  {mean:7.3f} +/- {std:5.3f} ms   "
          f"SEM {sem:.3f}   median {np.median(samples):7.3f}")
    return mean, std, samples


def main():
    print(
        f"Config: B={B} T={T} H={H} H_KV={H_KV} HEAD_DIM={HEAD_DIM} "
        f"ROPE_DIMS={ROPE_DIMS} N_LAYERS={N_LAYERS} dtype={DTYPE}\n"
        f"Forward + backward, fullgraph compile, dynamic=False, "
        f"{ITERS} iters after {WARMUP} warmup, fresh intradoc meta per iter.\n"
    )
    torch.manual_seed(0)
    x = torch.randn(B, T, DIM, dtype=DTYPE, device=DEVICE, requires_grad=False)

    base = BaselineAttn().to(DEVICE).to(DTYPE)
    lut = IntradocLutAttn().to(DEVICE).to(DTYPE)
    trig = IntradocTrigAttn().to(DEVICE).to(DTYPE)
    lut.load_state_dict(base.state_dict(), strict=False)
    trig.load_state_dict(base.state_dict(), strict=False)

    base_mean, _, base_samples = bench_fwd_bwd(
        '1. baseline (dense + cached)', base, x, intradoc=False
    )
    lut_mean, _, lut_samples = bench_fwd_bwd(
        '2. intradoc + LUT-gather RoPE', lut, x, intradoc=True
    )
    trig_mean, _, trig_samples = bench_fwd_bwd(
        '3. intradoc + trig-on-fly RoPE', trig, x, intradoc=True
    )

    # Welch's t-test to put the delta in context.
    from scipy.stats import ttest_ind
    print("\nDelta vs baseline (negative = faster):")
    for name, samples in [('LUT', lut_samples), ('trig', trig_samples)]:
        delta = samples.mean() - base_samples.mean()
        rel = delta / base_samples.mean() * 100
        try:
            t, p = ttest_ind(samples, base_samples, equal_var=False)
            print(f"  {name:6s} {delta:+.3f} ms  ({rel:+.2f}%)   "
                  f"Welch t={t:.2f}, p={p:.2e}")
        except Exception:
            print(f"  {name:6s} {delta:+.3f} ms  ({rel:+.2f}%)")


if __name__ == '__main__':
    main()
