#include "exception_handler.hpp"
#include "lib.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization report.
class DotProduct;

SYCL_EXTERNAL float dot_prod_16(sycl::float16 x, sycl::float16 y);

int main() {
#if defined(FPGA_EMULATOR)
    sycl::ext::intel::fpga_emulator_selector device_selector;
#else
    sycl::ext::intel::fpga_selector device_selector;
#endif

    try {
        sycl::queue q(device_selector, fpga_tools::exception_handler,
                      sycl::property::queue::enable_profiling());
        std::cout << "Device name: "
                  << q.get_device().get_info<sycl::info::device::name>().c_str()
                  << std::endl;

        float *X = (float *)malloc(num_elem_A * sizeof(float));
        float *Y = (float *)malloc(num_elem_B * sizeof(float));

        q..single_task<DotProduct>([=]() {
            // SyclSquare is a SYCL library function, defined in
            // lib_sycl.cpp.
            float a_sq = SyclSquare(kA);
            float b_sq = SyclSquare(kB);

            // RtlByteswap is an RTL library.
            //  - When compiled for FPGA, Verilog module byteswap_uint in
            //  lib_rtl.v
            //    is instantiated in the datapath by the compiler.
            //  - When compiled for FPGA emulator (CPU), the C model of
            //  RtlByteSwap
            //    in lib_rtl_model.cpp is used instead.
            accessor_c[0] = RtlByteswap((unsigned)(a_sq + b_sq));
        });

    } catch (sycl::exception const &e) {
        // Catches exceptions in the host code
        std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

        // Most likely the runtime couldn't find FPGA hardware!
        if (e.code().value() == CL_DEVICE_NOT_FOUND) {
            std::cerr
                << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
            std::cerr
                << "Run sys_check in the oneAPI root directory to verify.\n";
            std::cerr << "If you are targeting the FPGA emulator, compile with "
                         "-DFPGA_EMULATOR.\n";
        }
        std::terminate();
    }

    float kA = 2.0f;
    float kB = 3.0f;
    // Compute the expected "golden" result
    unsigned gold = (kA * kA) + (kB * kB);
    gold = gold << 16 | gold >> 16;

    // Check the results
    if (result != gold) {
        std::cout << "FAILED: result is incorrect!\n";
        return -1;
    }
    std::cout << "PASSED: result is correct!\n";
    return 0;
}