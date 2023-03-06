#include "operators.hpp"
#include "exception_handler.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#define LVEC 32768

using namespace sycl;

// Forward declare the kernel and pipe names
// This FPGA best practice reduces name mangling in the optimization report.
class XLoader;
class YLoader;
class DotProduct;
class ZUnloader;
class XPipe;
class YPipe;
class ZPipe;

SYCL_EXTERNAL float dot_prod_16(sycl::float16 x, sycl::float16 y);

constexpr size_t num_elem = LVEC;
constexpr size_t loop_iter = LVEC >> 4;

using XVecPipe = sycl::ext::intel::pipe<XPipe, float16, 256>;
using YVecPipe = sycl::ext::intel::pipe<YPipe, float16, 256>;

int main() {
#if defined(FPGA_EMULATOR)
    sycl::ext::intel::fpga_emulator_selector device_selector;
#else
    sycl::ext::intel::fpga_selector device_selector;
#endif

    std::vector<sycl::event> kernel_events;
    sycl::queue q(device_selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling());
    std::cout << "Device name: "
              << q.get_device().get_info<sycl::info::device::name>().c_str()
              << std::endl;

    float *X = (float *)malloc(num_elem * sizeof(float));
    float *Y = (float *)malloc(num_elem * sizeof(float));
    float result, gold = 0.0f;

    for (size_t i = 0; i < num_elem; i++) {
        X[i] = random();
        Y[i] = random();
        gold += X[i] * Y[i];
    }

    float *X_device = sycl::malloc_device<float>(num_elem, q);
    float *Y_device = sycl::malloc_device<float>(num_elem, q);
    float *Z_device = sycl::malloc_device<float>(1, q);

    q.memcpy(X_device, X, num_elem * sizeof(float)).wait();
    q.memcpy(Y_device, Y, num_elem * sizeof(float)).wait();

    kernel_events.push_back(
        q.single_task<XLoader>([=]() [[intel::kernel_args_restrict]] {
            sycl::device_ptr<float> X_d(X_device);
            for (int i = 0; i < loop_iter; i++) {
                float16 ddr_read;
#pragma unroll
                for (int vec_idx = 0; vec_idx < 16; vec_idx++) {
                    ddr_read[vec_idx] = X_d[(i << 4) + vec_idx];
                }
                XVecPipe::write(ddr_read);
            }
        }));

    kernel_events.push_back(
        q.single_task<YLoader>([=]() [[intel::kernel_args_restrict]] {
            sycl::device_ptr<float> Y_d(Y_device);
            for (int i = 0; i < loop_iter; i++) {
                float16 ddr_read;
#pragma unroll
                for (int vec_idx = 0; vec_idx < 16; vec_idx++) {
                    ddr_read[vec_idx] = Y_d[(i << 4) + vec_idx];
                }
                YVecPipe::write(ddr_read);
            }
        }));

    kernel_events.push_back(
        q.single_task<DotProduct>([=]() [[intel::kernel_args_restrict]] {
            sycl::device_ptr<float> Z_d(Z_device);
            float acc = 0.0f;
            for (int i = 0; i < loop_iter; i++) {
                float16 x_vec = XVecPipe::read();
                float16 y_vec = YVecPipe::read();
                float partial_acc = dot_prod_16(x_vec, y_vec);
                acc += partial_acc;
            }
            Z_d[0] = acc;
        }));

    for (unsigned int i = 0; i < kernel_events.size(); i++) {
        kernel_events.at(i).wait();
    }

    if (kernel_events.size() > 0) {
        double k_earliest_start_time =
            kernel_events.at(0)
                .get_profiling_info<
                    sycl::info::event_profiling::command_start>();
        double k_latest_end_time =
            kernel_events.at(0)
                .get_profiling_info<sycl::info::event_profiling::command_end>();
        for (unsigned i = 1; i < kernel_events.size(); i++) {
            double tmp_start =
                kernel_events.at(i)
                    .get_profiling_info<
                        sycl::info::event_profiling::command_start>();
            double tmp_end =
                kernel_events.at(i)
                    .get_profiling_info<
                        sycl::info::event_profiling::command_end>();
            if (tmp_start < k_earliest_start_time) {
                k_earliest_start_time = tmp_start;
            }
            if (tmp_end > k_latest_end_time) {
                k_latest_end_time = tmp_end;
            }
        }
        // Get time in ns
        double events_time = (k_latest_end_time - k_earliest_start_time);
        printf("  Time: %.5f ns\n", events_time);
        printf("  Throughput: %.2f GFLOPS\n", (double)2.0 * LVEC / events_time);
    }

    q.memcpy(&result, Z_device, sizeof(float)).wait();

    // Check the results
    if (fabs(result - gold) > 0.005*fabs(gold)) {
        std::cout << "result: " << result << ", gold: " << gold << std::endl;
        std::cout << "FAILED: result is incorrect!" << std::endl;
        return -1;
    }
    std::cout << "PASSED: result is correct!" << std::endl;
    return 0;
}
