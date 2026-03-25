#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void crt_kernel(const int *__restrict__ residues, const double *__restrict__ wh,
                           const double *__restrict__ wm, const double *__restrict__ wl,
                           double M_hi, double M_lo, double inv_M, double scale_sq,
                           const double *__restrict__ row_max, const double *__restrict__ col_max,
                           double *__restrict__ out, int n_elems, int cols, int n_mod) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elems)
        return;

    double acc_hi = 0.0, acc_lo = 0.0;

    for (int k = 0; k < n_mod; k++) {
        double r = (double)residues[k * n_elems + idx];

        // triple-float CRT accumulation
        double prod = r * wh[k];
        double e = fma(r, wh[k], -prod);

        double s = acc_hi + prod;
        double v = s - acc_hi;
        double err = (acc_hi - (s - v)) + (prod - v);
        acc_hi = s;
        acc_lo = acc_lo + fma(r, wm[k], e) + r * wl[k] + err;
    }

    // symmetric modulo
    double q = rint((acc_hi + acc_lo) * inv_M);
    double pm = q * M_hi;
    double em = fma(q, M_hi, -pm);
    double result = (acc_hi - pm) + (acc_lo - em - q * M_lo);

    int i = idx / cols;
    int j = idx % cols;
    out[idx] = result / scale_sq * row_max[i] * col_max[j];
}

torch::Tensor crt_reconstruct(torch::Tensor residues, torch::Tensor wh, torch::Tensor wm,
                              torch::Tensor wl, double M_hi, double M_lo, double inv_M,
                              double scale_sq, torch::Tensor row_max, torch::Tensor col_max,
                              int rows, int cols, int n_mod) {
    int n_elems = rows * cols;
    auto out = torch::empty({rows, cols}, torch::dtype(torch::kFloat64).device(residues.device()));

    int threads = 256;
    int blocks = (n_elems + threads - 1) / threads;

    crt_kernel<<<blocks, threads>>>(
        residues.data_ptr<int>(), wh.data_ptr<double>(), wm.data_ptr<double>(),
        wl.data_ptr<double>(), M_hi, M_lo, inv_M, scale_sq, row_max.data_ptr<double>(),
        col_max.data_ptr<double>(), out.data_ptr<double>(), n_elems, cols, n_mod);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("crt_reconstruct", &crt_reconstruct, "CRT reconstruction via triple-float FMA");
}
