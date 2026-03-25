import math

import torch

_SCALE = float(2**52)

# fmt: off
_PRIMES = [
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
]
# fmt: on


def _n_moduli(k):
    """min primes whose product covers dot-product range"""
    target = 2 * k * _SCALE * _SCALE
    prod = 1
    for i, p in enumerate(_PRIMES):
        prod *= p
        if prod > target:
            return i + 1
    raise ValueError(f"not enough primes for inner dim {k}")


def _scale_to_int(A, B):
    """row/col normalize, scale to int52"""
    row_max = A.abs().amax(dim=-1).clamp(min=1e-300)
    col_max = B.abs().amax(dim=-2).clamp(min=1e-300)
    A_int = torch.round(A / row_max.unsqueeze(-1) * _SCALE).to(torch.int64)
    B_int = torch.round(B / col_max.unsqueeze(-2) * _SCALE).to(torch.int64)
    return A_int, B_int, row_max, col_max


def _residues(X_int, moduli):
    """int64 -> stacked int8 residues per prime"""
    out = torch.empty(len(moduli), *X_int.shape, dtype=torch.int8, device=X_int.device)
    for i, m in enumerate(moduli):
        out[i] = (X_int % m).to(torch.int8)
    return out


def _matmul_residues(a_res, b_res, moduli):
    """per-prime matmul -> modular products"""
    use_float = a_res.is_cuda
    residues = []
    for i, m in enumerate(moduli):
        if use_float:
            c = (a_res[i].float() @ b_res[i].float()).to(torch.int32)
        else:
            c = a_res[i].int() @ b_res[i].int()
        residues.append(c % m)
    return residues


def _crt_precompute(moduli):
    """CRT coefficients as Python bigints (exact)"""
    M = math.prod(moduli)
    coeffs = []
    for m in moduli:
        Mi = M // m
        coeffs.append(Mi * pow(Mi, -1, m))
    return coeffs, M


def _crt_reconstruct(residues, moduli, scale_sq, row_max, col_max):
    """CRT reconstruction via Python bigints"""
    device = residues[0].device
    rows, cols = residues[0].shape
    coeffs, M = _crt_precompute(moduli)
    half_M = M // 2

    res_lists = [r.cpu().tolist() for r in residues]
    row_s = row_max.cpu().tolist()
    col_s = col_max.cpu().tolist()

    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            x = sum(res_lists[k][i][j] * coeffs[k] for k in range(len(moduli))) % M
            if x > half_M:
                x -= M
            out[i][j] = x / scale_sq * row_s[i] * col_s[j]

    return torch.tensor(out, dtype=torch.float64, device=device)


def crt_matmul(A, B):
    """FP64-accurate matmul via int8 CRT"""
    n_mod = _n_moduli(A.shape[-1])
    moduli = _PRIMES[:n_mod]
    scale_sq = _SCALE**2

    A_int, B_int, row_max, col_max = _scale_to_int(A, B)
    a_res = _residues(A_int, moduli)
    b_res = _residues(B_int, moduli)
    residues = _matmul_residues(a_res, b_res, moduli)

    if A.is_cuda:
        from tensor_inv._cuda_crt import cuda_crt_reconstruct

        return cuda_crt_reconstruct(residues, moduli, scale_sq, row_max, col_max)

    return _crt_reconstruct(residues, moduli, scale_sq, row_max, col_max)
