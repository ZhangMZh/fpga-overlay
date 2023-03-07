#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

SYCL_EXTERNAL float dot_prod_16(sycl::float16 x, sycl::float16 y) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

// template <typename T, int N>
// SYCL_EXTERNAL T dot_prod(sycl::vec<T, N> x, sycl::vec<T, N> y) {
//     T sum = (T)0;
// #pragma unroll
//     for (int i = 0; i < N; i++) {
//         sum += x[i] * y[i];
//     }
//     return sum;
// }
