#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Batched int8 GEMM via cuBLAS + modular reduction

__global__ void mod_reduce(int32_t *data, const int32_t *moduli, int elems_per_batch, int n_batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elems_per_batch * n_batch)
        return;
    int batch = idx / elems_per_batch;
    int32_t m = moduli[batch];
    int32_t v = data[idx] % m;
    if (v < 0)
        v += m;
    data[idx] = v;
}

torch::Tensor batched_int8_gemm_mod(torch::Tensor a, torch::Tensor b, torch::Tensor moduli) {
    int batch = a.size(0);
    int M = a.size(1);
    int K = a.size(2);
    int N = b.size(2);

    auto c = torch::empty({batch, M, N}, torch::dtype(torch::kInt32).device(a.device()));

    int32_t alpha = 1, beta = 0;
    auto handle = at::cuda::getCurrentCUDABlasHandle();

    // row-major A(M,K) @ B(K,N) = C(M,N)
    // cuBLAS col-major: C^T(N,M) = B^T(N,K) @ A^T(K,M)
    TORCH_CHECK(cublasGemmStridedBatchedEx(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b.data_ptr<int8_t>(),
                    CUDA_R_8I, N, (long long)K * N, a.data_ptr<int8_t>(), CUDA_R_8I, K,
                    (long long)M * K, &beta, c.data_ptr<int32_t>(), CUDA_R_32I, N, (long long)M * N,
                    batch, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS,
                "cuBLAS batched int8 GEMM failed");

    // in-place modular reduction
    int total = batch * M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mod_reduce<<<blocks, threads>>>(c.data_ptr<int32_t>(), moduli.data_ptr<int32_t>(), M * N,
                                    batch);

    return c;
}

// CRT reconstruction via triple-float FMA
__global__ void crt_kernel(const int *__restrict__ residues, const double *__restrict__ wh,
                           const double *__restrict__ wm, const double *__restrict__ wl,
                           double M_hi, double M_lo, double inv_M, const int *__restrict__ row_exp,
                           const int *__restrict__ col_exp, int two_bits, double *__restrict__ out,
                           int n_elems, int cols, int n_mod) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elems)
        return;

    double acc_hi = 0.0, acc_lo = 0.0;

    for (int k = 0; k < n_mod; k++) {
        double r = (double)residues[k * n_elems + idx];

        double prod = r * wh[k];
        double e = fma(r, wh[k], -prod);

        double s = acc_hi + prod;
        double v = s - acc_hi;
        double err = (acc_hi - (s - v)) + (prod - v);
        acc_hi = s;
        acc_lo = acc_lo + fma(r, wm[k], e) + r * wl[k] + err;
    }

    double q = rint((acc_hi + acc_lo) * inv_M);
    double pm = q * M_hi;
    double em = fma(q, M_hi, -pm);
    double result = (acc_hi - pm) + (acc_lo - em - q * M_lo);

    int i = idx / cols;
    int j = idx % cols;
    out[idx] = ldexp(result, row_exp[i] + col_exp[j] - two_bits);
}

torch::Tensor crt_reconstruct(torch::Tensor residues, torch::Tensor wh, torch::Tensor wm,
                              torch::Tensor wl, double M_hi, double M_lo, double inv_M,
                              torch::Tensor row_exp, torch::Tensor col_exp, int two_bits, int rows,
                              int cols, int n_mod) {
    int n_elems = rows * cols;
    auto out = torch::empty({rows, cols}, torch::dtype(torch::kFloat64).device(residues.device()));

    int threads = 256;
    int blocks = (n_elems + threads - 1) / threads;

    crt_kernel<<<blocks, threads>>>(residues.data_ptr<int>(), wh.data_ptr<double>(),
                                    wm.data_ptr<double>(), wl.data_ptr<double>(), M_hi, M_lo, inv_M,
                                    row_exp.data_ptr<int>(), col_exp.data_ptr<int>(), two_bits,
                                    out.data_ptr<double>(), n_elems, cols, n_mod);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("crt_reconstruct", &crt_reconstruct, "CRT reconstruction via triple-float FMA");
    m.def("batched_int8_gemm_mod", &batched_int8_gemm_mod, "Batched int8 GEMM + modular reduction");
}
