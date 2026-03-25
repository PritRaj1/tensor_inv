import math

import torch
import triton
import triton.language as tl


def _crt_weights(moduli):
    """CRT weights as triple-float (hi, mid, lo)"""
    M = math.prod(moduli)
    wh, wm, wl = [], [], []
    for m in moduli:
        Mi = M // m
        w = Mi * pow(Mi, -1, m)
        hi = float(w)
        r = w - int(hi)
        mid = float(r)
        lo = float(r - int(mid))
        wh.append(hi)
        wm.append(mid)
        wl.append(lo)

    M_hi = float(M)
    M_lo = float(M - int(M_hi))
    inv_M = 1.0 / float(M)
    return wh, wm, wl, M_hi, M_lo, inv_M


@triton.jit
def _fused_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    row_max_ptr,
    col_max_ptr,
    wh_ptr,
    wm_ptr,
    wl_ptr,
    const_ptr,
    primes_ptr,
    M,
    N,
    K,
    n_mod: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)

    MK = M * K
    KN = K * N

    acc_hi = tl.zeros([BM, BN], dtype=tl.float64)
    acc_lo = tl.zeros([BM, BN], dtype=tl.float64)

    for i in tl.static_range(n_mod):
        c = tl.zeros([BM, BN], dtype=tl.int32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            a = tl.load(
                a_ptr + i * MK + rm[:, None] * K + rk[None, :],
                mask=(rm[:, None] < M) & (rk[None, :] < K),
                other=0,
            )
            b = tl.load(
                b_ptr + i * KN + rk[:, None] * N + rn[None, :],
                mask=(rk[:, None] < K) & (rn[None, :] < N),
                other=0,
            )
            c += tl.dot(a, b)

        # mod p, ensure non-negative
        p = tl.load(primes_ptr + i).to(tl.int32)
        r = c % p
        r = tl.where(r < 0, r + p, r)

        # CRT accumulate (triple-float FMA)
        r_f = r.to(tl.float64)
        w_hi = tl.load(wh_ptr + i).to(tl.float64)
        w_mid = tl.load(wm_ptr + i).to(tl.float64)
        w_lo = tl.load(wl_ptr + i).to(tl.float64)

        prod = r_f * w_hi
        e = tl.fma(r_f, w_hi, -prod)

        s = acc_hi + prod
        v = s - acc_hi
        err = (acc_hi - (s - v)) + (prod - v)
        acc_hi = s
        acc_lo = acc_lo + tl.fma(r_f, w_mid, e) + r_f * w_lo + err

    # symmetric modulo
    M_hi = tl.load(const_ptr + 0).to(tl.float64)
    M_lo = tl.load(const_ptr + 1).to(tl.float64)
    inv_M = tl.load(const_ptr + 2).to(tl.float64)
    scale_sq = tl.load(const_ptr + 3).to(tl.float64)

    q = tl.extra.cuda.libdevice.rint((acc_hi + acc_lo) * inv_M)
    pm = q * M_hi
    em = tl.fma(q, M_hi, -pm)
    result = (acc_hi - pm) + (acc_lo - em - q * M_lo)

    mask_out = (rm[:, None] < M) & (rn[None, :] < N)
    row_s = tl.load(row_max_ptr + rm, mask=rm < M, other=1.0)
    col_s = tl.load(col_max_ptr + rn, mask=rn < N, other=1.0)
    result = result / scale_sq * row_s[:, None] * col_s[None, :]

    tl.store(out_ptr + rm[:, None] * N + rn[None, :], result, mask=mask_out)


def fused_ozaki_matmul(a_res, b_res, moduli, scale_sq, row_max, col_max):
    n_mod, M, K = a_res.shape
    N = b_res.shape[2]
    device = a_res.device

    wh, wm, wl, M_hi, M_lo, inv_M = _crt_weights(moduli)
    wh_t = torch.tensor(wh, dtype=torch.float64, device=device)
    wm_t = torch.tensor(wm, dtype=torch.float64, device=device)
    wl_t = torch.tensor(wl, dtype=torch.float64, device=device)
    const = torch.tensor(
        [M_hi, M_lo, inv_M, scale_sq], dtype=torch.float64, device=device
    )
    primes = torch.tensor(moduli, dtype=torch.int32, device=device)
    out = torch.empty(M, N, dtype=torch.float64, device=device)

    BM, BN, BK = 32, 32, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    _fused_kernel[grid](
        a_res,
        b_res,
        out,
        row_max,
        col_max,
        wh_t,
        wm_t,
        wl_t,
        const,
        primes,
        M,
        N,
        K,
        n_mod=n_mod,
        BM=BM,
        BN=BN,
        BK=BK,
        enable_fp_fusion=False,
    )

    return out
