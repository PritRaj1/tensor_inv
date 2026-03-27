import math

import torch

# fmt: off
_PRIMES = [
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
]
# fmt: on

_PRECISION_BITS = {
    torch.float16: 11,
    torch.bfloat16: 8,
    torch.float32: 24,
    torch.float64: 53,
}


def _n_moduli(k, bits):
    """min primes whose product covers dot-product range 2*k*2^(2*bits)"""
    target = 2 * k * (2 ** (2 * bits))
    prod = 1
    for i, p in enumerate(_PRIMES):
        prod *= p
        if prod > target:
            return i + 1

    raise ValueError(f"not enough primes for inner dim {k}, bits {bits}")


def _scale_to_int(A, B, bits):
    """power-of-2 diagonal scaling (exact in FP), then round to int"""
    row_exp = A.abs().amax(dim=-1).clamp(min=1e-300).frexp()[1]
    col_exp = B.abs().amax(dim=-2).clamp(min=1e-300).frexp()[1]
    A_int = torch.ldexp(A, bits - row_exp.unsqueeze(-1)).round().to(torch.int64)
    B_int = torch.ldexp(B, bits - col_exp.unsqueeze(-2)).round().to(torch.int64)
    return A_int, B_int, row_exp, col_exp


def _residues(X_int, moduli):
    """int64 -> stacked int8 residues per prime"""
    moduli_t = torch.tensor(moduli, dtype=torch.int64, device=X_int.device).view(
        -1, *([1] * X_int.ndim)
    )
    return (X_int.unsqueeze(0) % moduli_t).to(torch.int8)


def _pad8(x):
    """pad last two dims to multiples of 8 for _int_mm"""
    m, n = x.shape[-2], x.shape[-1]
    pm = (-m) % 8
    pn = (-n) % 8
    if pm == 0 and pn == 0:
        return x
    return torch.nn.functional.pad(x, (0, pn, 0, pm))


def _matmul_residues_impl(a_res, b_res, moduli_t):
    """per-prime int8 matmul + modular reduction"""
    m, n = a_res.shape[-2], b_res.shape[-1]
    a_p = _pad8(a_res)
    b_p = _pad8(b_res)
    results = []
    for i in range(a_p.shape[0]):
        c = torch._int_mm(a_p[i], b_p[i])[:m, :n]
        results.append(c % moduli_t[i])
    return results


_matmul_residues_cuda = torch.compile(_matmul_residues_impl, dynamic=True)


def _matmul_residues(a_res, b_res, moduli):
    """per-prime matmul -> modular products"""
    moduli_t = torch.tensor(moduli, dtype=torch.int32, device=a_res.device)
    if a_res.is_cuda:
        return _matmul_residues_cuda(a_res, b_res, moduli_t)
    c = a_res.int() @ b_res.int()
    return list(c % moduli_t.view(-1, 1, 1))


def _crt_precompute(moduli):
    """CRT coefficients as Python bigints (exact)"""
    M = math.prod(moduli)
    coeffs = []
    for m in moduli:
        Mi = M // m
        coeffs.append(Mi * pow(Mi, -1, m))

    return coeffs, M


def _crt_reconstruct(residues, moduli, bits, row_exp, col_exp):
    """CRT reconstruction via Python bigints + ldexp unscaling"""
    device = residues[0].device
    rows, cols = residues[0].shape
    coeffs, M = _crt_precompute(moduli)
    half_M = M // 2

    res_lists = [r.cpu().tolist() for r in residues]
    row_e = row_exp.cpu().tolist()
    col_e = col_exp.cpu().tolist()

    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            x = sum(res_lists[k][i][j] * coeffs[k] for k in range(len(moduli))) % M
            if x > half_M:
                x -= M
            out[i][j] = math.ldexp(float(x), row_e[i] + col_e[j] - 2 * bits)

    return torch.tensor(out, dtype=torch.float64, device=device)


def crt_matmul(A, B):
    """FP-accurate matmul via int8 CRT. Precision matches input dtype."""
    bits = _PRECISION_BITS.get(A.dtype)
    if bits is None:
        supported = ", ".join(str(d) for d in _PRECISION_BITS)
        raise TypeError(
            f"unsupported dtype {A.dtype}. "
            f"CRT matmul emulates floating-point precision via int8 arithmetic; "
            f"input must be a float type ({supported})"
        )

    A, B = A.double(), B.double()
    n_mod = _n_moduli(A.shape[-1], bits)
    moduli = _PRIMES[:n_mod]

    A_int, B_int, row_exp, col_exp = _scale_to_int(A, B, bits)
    a_res = _residues(A_int, moduli)
    b_res = _residues(B_int, moduli)
    residues = _matmul_residues(a_res, b_res, moduli)

    if A.is_cuda:
        from tensor_inv._cuda_crt import cuda_crt_reconstruct

        return cuda_crt_reconstruct(residues, moduli, bits, row_exp, col_exp)

    return _crt_reconstruct(residues, moduli, bits, row_exp, col_exp)
