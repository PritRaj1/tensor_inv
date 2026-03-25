import math
import warnings

import torch

_SCALE = float(2**52)

# fmt: off
_PRIMES = [
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
]
# fmt: on

_HAS_TENSOR_CORES = None


def _check_tensor_cores():
    global _HAS_TENSOR_CORES
    if _HAS_TENSOR_CORES is None:
        if not torch.cuda.is_available():
            _HAS_TENSOR_CORES = False
        else:
            cap = torch.cuda.get_device_capability()
            name = torch.cuda.get_device_name(0).lower()
            _HAS_TENSOR_CORES = cap >= (7, 5) and "gtx 16" not in name
    return _HAS_TENSOR_CORES


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
    """row/col normalize, scale to int64"""
    row_max = A.abs().amax(dim=1).clamp(min=1e-300)
    col_max = B.abs().amax(dim=0).clamp(min=1e-300)
    A_int = torch.round(A / row_max.unsqueeze(1) * _SCALE).to(torch.int64)
    B_int = torch.round(B / col_max.unsqueeze(0) * _SCALE).to(torch.int64)
    return A_int, B_int, row_max, col_max


def _int8_residues(X_int, moduli):
    """int64 -> stacked int8 residues per prime"""
    out = torch.empty(len(moduli), *X_int.shape, dtype=torch.int8, device=X_int.device)
    for i, m in enumerate(moduli):
        out[i] = (X_int % m).to(torch.int8)
    return out


def _matmul_residues(a_res, b_res, moduli):
    """per-prime matmul, fallback when no tensor cores"""
    use_float = a_res.is_cuda
    residues = []
    for i, m in enumerate(moduli):
        if use_float:
            c = (a_res[i].float() @ b_res[i].float()).to(torch.int32)
        else:
            c = a_res[i].int() @ b_res[i].int()

        residues.append(c % m)

    return residues


def _crt_scalar(residues, moduli):
    """exact CRT via Python bigints"""
    M = math.prod(moduli)
    x = 0
    for r, m in zip(residues, moduli):
        Mi = M // m
        x += r * Mi * pow(Mi, -1, m)
    x %= M
    return x - M if x > M // 2 else x


def _crt_cpu(residues, moduli, scale_sq, row_max, col_max):
    """element-wise CRT reconstruction on CPU"""
    M, N = residues[0].shape
    out = torch.empty(M, N, dtype=torch.float64)
    for i in range(M):
        for j in range(N):
            rs = [residues[k][i, j].item() for k in range(len(moduli))]
            out[i, j] = (
                _crt_scalar(rs, moduli)
                / scale_sq
                * row_max[i].item()
                * col_max[j].item()
            )
    return out


def _forward(A, B):
    n_mod = _n_moduli(A.shape[1])
    moduli = _PRIMES[:n_mod]
    scale_sq = _SCALE**2

    A_int, B_int, row_max, col_max = _scale_to_int(A, B)
    a_res = _int8_residues(A_int, moduli)
    b_res = _int8_residues(B_int, moduli)

    # fused: tensor cores do matmul + CRT in one kernel
    if A.is_cuda and _check_tensor_cores():
        from tensor_inv._triton_matmul import fused_ozaki_matmul
        return fused_ozaki_matmul(a_res, b_res, moduli, scale_sq, row_max, col_max)

    # unfused: per-prime matmul, then CRT
    if A.is_cuda:
        warnings.warn(
            "no INT8 tensor cores detected, using slower fallback",
            stacklevel=3,
        )

    residues = _matmul_residues(a_res, b_res, moduli)

    if A.is_cuda:
        from tensor_inv._triton_matmul import crt_reconstruct
        return crt_reconstruct(residues, moduli, scale_sq, row_max, col_max)

    return _crt_cpu(residues, moduli, scale_sq, row_max, col_max)


class _OzakiMatmul(torch.autograd.Function):
    @staticmethod
    def forward(A, B):
        return _forward(A, B)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad):
        A, B = ctx.saved_tensors
        grad_A = _forward(grad, B.T) if ctx.needs_input_grad[0] else None
        grad_B = _forward(A.T, grad) if ctx.needs_input_grad[1] else None
        return grad_A, grad_B


def ozaki_matmul(A, B):
    """fp64-accurate matmul via INT8 tensor cores, with autograd"""
    return _OzakiMatmul.apply(A, B)
