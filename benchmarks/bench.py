import torch
import pickle
from torch.utils.benchmark import Compare, Timer

from tensor_inv import cholesky, crt_matmul, lu, rsvd

SIZES = [64, 128, 256, 512, 1024]


def make_timers(device, dtype=torch.float64):
    timers = []

    for n in SIZES:
        shared = {"dtype": dtype, "device": device, "n": n, "torch": torch}

        # matmul
        setup = f"A = torch.randn({n},{n},dtype=dtype,device=device); B = torch.randn({n},{n},dtype=dtype,device=device)"
        timers.append(
            Timer(
                "A @ B",
                setup,
                globals=shared,
                label="matmul",
                sub_label=f"n={n}",
                description="torch",
            )
        )
        timers.append(
            Timer(
                "crt_matmul(A, B)",
                setup,
                globals={**shared, "crt_matmul": crt_matmul},
                label="matmul",
                sub_label=f"n={n}",
                description="CRT",
            )
        )

        # cholesky
        setup_spd = f"M = torch.randn({n},{n},dtype=dtype,device=device); A = M @ M.T + {n}*torch.eye({n},dtype=dtype,device=device)"
        timers.append(
            Timer(
                "torch.linalg.cholesky(A)",
                setup_spd,
                globals=shared,
                label="cholesky",
                sub_label=f"n={n}",
                description="torch",
            )
        )
        timers.append(
            Timer(
                "cholesky(A)",
                setup_spd,
                globals={**shared, "cholesky": cholesky},
                label="cholesky",
                sub_label=f"n={n}",
                description="CRT",
            )
        )

        # lu
        setup_sq = f"A = torch.randn({n},{n},dtype=dtype,device=device)"
        timers.append(
            Timer(
                "torch.linalg.lu(A)",
                setup_sq,
                globals=shared,
                label="lu",
                sub_label=f"n={n}",
                description="torch",
            )
        )
        timers.append(
            Timer(
                "lu(A)",
                setup_sq,
                globals={**shared, "lu": lu},
                label="lu",
                sub_label=f"n={n}",
                description="CRT",
            )
        )

        # svd vs rsvd
        timers.append(
            Timer(
                "torch.linalg.svd(A, full_matrices=False)",
                setup_sq,
                globals=shared,
                label="svd",
                sub_label=f"n={n}",
                description="torch (full)",
            )
        )
        timers.append(
            Timer(
                "rsvd(A, rank=20)",
                setup_sq,
                globals={**shared, "rsvd": rsvd},
                label="svd",
                sub_label=f"n={n}",
                description="CRT (rank=20)",
            )
        )

    return timers


def run(device):
    print(f"\n{'=' * 60}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    else:
        print(f"device: {device}")
    print(f"{'=' * 60}")

    timers = make_timers(device)
    results = [t.blocked_autorange(min_run_time=1.0) for t in timers]
    compare = Compare(results)
    compare.print()
    return results


if __name__ == "__main__":
    all_results = []
    all_results.extend(run("cpu"))

    if torch.cuda.is_available():
        all_results.extend(run("cuda"))

    with open("benchmarks/results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print("\nresults saved to benchmarks/results.pkl")
