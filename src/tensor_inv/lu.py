import torch

from tensor_inv.crt_matmul import crt_matmul


def lu(A, block_size=64):
    """
    Blocked LU with partial pivoting: panel via torch, trailing GEMM via CRT int8.

    Returns P, L, U such that PA = LU.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"expected square matrix, got {A.shape}")

    n = A.shape[0]
    W = A.clone()
    piv = torch.arange(n, device=A.device)

    for j in range(0, n, block_size):
        b = min(block_size, n - j)
        jb = j + b

        # panel: unblocked LU with partial pivoting
        for k in range(j, jb):
            max_idx = W[k:, k].abs().argmax() + k
            if max_idx != k:
                W[k], W[max_idx] = W[max_idx].clone(), W[k].clone()
                piv[k], piv[max_idx] = piv[max_idx].clone(), piv[k].clone()

            if W[k, k].abs() > 1e-15:
                W[k + 1 :, k] /= W[k, k]
                W[k + 1 :, k + 1 : jb] -= (
                    W[k + 1 :, k : k + 1] @ W[k : k + 1, k + 1 : jb]
                )

        if jb < n:
            # triangular solve for U block row
            L_panel = torch.tril(W[j:jb, j:jb], diagonal=-1) + torch.eye(
                b, dtype=A.dtype, device=A.device
            )
            W[j:jb, jb:] = torch.linalg.solve_triangular(
                L_panel, W[j:jb, jb:], upper=False
            )

            # trailing GEMM update via CRT (int8 tensor cores)
            W[jb:, jb:] -= crt_matmul(W[jb:, j:jb], W[j:jb, jb:])

    L = torch.tril(W, diagonal=-1) + torch.eye(n, dtype=A.dtype, device=A.device)
    U = torch.triu(W)

    P = torch.zeros(n, n, dtype=A.dtype, device=A.device)
    P[torch.arange(n, device=A.device), piv] = 1.0

    return P, L, U
