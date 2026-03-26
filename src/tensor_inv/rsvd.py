import torch

from tensor_inv.crt_matmul import crt_matmul


def rsvd(A, rank, n_oversamples=10, n_power_iter=1):
    """
    Randomized SVD via CRT int8 GEMM (Halko-Martinsson-Tropp).

    Returns U (m x rank), S (rank,), Vt (rank x n).
    """
    m, n = A.shape
    r = rank + n_oversamples
    dtype = A.dtype
    device = A.device

    # random projection
    Omega = torch.randn(n, r, dtype=dtype, device=device)
    Y = crt_matmul(A, Omega)
    Q, _ = torch.linalg.qr(Y)

    # power iteration!
    for _ in range(n_power_iter):
        Y = crt_matmul(A, crt_matmul(A.T, Q))
        Q, _ = torch.linalg.qr(Y)

    # project to low-rank subspace
    B = crt_matmul(Q.T, A)
    Ub, S, Vt = torch.linalg.svd(B, full_matrices=False)
    U = crt_matmul(Q, Ub)

    return U[:, :rank], S[:rank], Vt[:rank]
