import torch

from tensor_inv.crt_matmul import crt_matmul


def cholesky(A, block_size=64):
    """
    Blocked Cholesky: panel via torch, trailing GEMM via CRT int8.

    A must be symmetric positive definite.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"expected square matrix, got {A.shape}")

    n = A.shape[0]
    L = torch.zeros_like(A)
    W = A.clone()

    for j in range(0, n, block_size):
        b = min(block_size, n - j)
        jb = j + b

        # panel: small Cholesky on diagonal block (e.g for VPU)
        L[j:jb, j:jb] = torch.linalg.cholesky(W[j:jb, j:jb])

        if jb < n:
            # panel: triangular solve for sub-diagonal block (for VPU)
            L[jb:, j:jb] = torch.linalg.solve_triangular(
                L[j:jb, j:jb], W[jb:, j:jb].T, upper=False
            ).T

            # trailing GEMM update via CRT (int8 tensor cores)
            W[jb:, jb:] -= crt_matmul(L[jb:, j:jb], L[jb:, j:jb].T)

    return L
