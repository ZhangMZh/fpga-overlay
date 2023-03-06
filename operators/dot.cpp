#include <sycl/sycl.hpp>

SYCL_EXTERNAL extern "C" float dot_prod_16(sycl::float16 x, sycl::float16 y) {
    float sum == 0.0f;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        sum += x[i] * y[i];
        if ((i % 4) == 3) {
            sum = __fpga_reg(sum);
        }
    }
    return sum;
}