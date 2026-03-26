import torch

from tensor_inv.rsvd import rsvd


def test_rsvd_low_rank():
    torch.manual_seed(42)
    m, n, r = 100, 80, 10
    A = torch.randn(m, r, dtype=torch.float64) @ torch.randn(r, n, dtype=torch.float64)
    A += 1e-6 * torch.randn(m, n, dtype=torch.float64)

    U, S, Vt = rsvd(A, rank=r)
    A_approx = U @ torch.diag(S) @ Vt
    rel_err = (A - A_approx).norm() / A.norm()
    assert rel_err < 1e-4, f"rel err {rel_err:.2e}"


def test_rsvd_vs_torch():
    torch.manual_seed(0)
    A = torch.randn(64, 48, dtype=torch.float64)
    r = 10

    U, S, Vt = rsvd(A, rank=r)
    _, S_ref, _ = torch.linalg.svd(A, full_matrices=False)

    # top-r singular values should be close
    rel_err = (S - S_ref[:r]).abs().max() / S_ref[0]
    assert rel_err < 0.1, f"singular value err {rel_err:.2e}"
