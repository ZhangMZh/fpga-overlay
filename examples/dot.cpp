#include "data_transfer.hpp"
#include "exception_handler.hpp"
#include "operators.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#define LVEC 33554432 // 2^25

using namespace sycl;

// Forward declare the kernel and pipe names
// This FPGA best practice reduces name mangling in the optimization report.
class XLoader;
class YLoader;
class DotProduct;
class XPipe;
class YPipe;

constexpr size_t num_elems = LVEC;
constexpr size_t loop_iter = LVEC >> 4;

using XVecPipe = sycl::ext::intel::pipe<XPipe, sycl::vec<float, 16>, 256>;
using YVecPipe = sycl::ext::intel::pipe<YPipe, sycl::vec<float, 16>, 256>;
using BurstCoalescedLSU = sycl::ext::intel::lsu<sycl::ext::intel::burst_coalesce<true>,
                                                sycl::ext::intel::statically_coalesce<false>>;

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

    float *X = (float *)malloc(num_elems * sizeof(float));
    float *Y = (float *)malloc(num_elems * sizeof(float));
    float result, gold = 0.0f;

    for (size_t i = 0; i < num_elems; i++) {
        X[i] = (float)(rand() % 256) / 256.0f;
        Y[i] = (float)(rand() % 256) / 256.0f;
        gold += X[i] * Y[i];
    }

    {
        sycl::buffer<float> X_buf(X, num_elems, {sycl::property::buffer::mem_channel{1}});
        sycl::buffer<float> Y_buf(Y, num_elems, {sycl::property::buffer::mem_channel{2}});
        sycl::buffer<float> Z_buf(&result, 1);

        kernel_events.push_back(q.submit([&](handler &h) {
            sycl::accessor aX(X_buf, h, sycl::read_only);

            h.single_task<XLoader>([=]() [[intel::kernel_args_restrict]] {
                auto aX_ptr = aX.get_pointer();
                for (int i = 0; i < loop_iter; i++) {
                    float16 ddr_read;
#pragma unroll
                    for (int vec_idx = 0; vec_idx < 16; vec_idx++) {
                        ddr_read[vec_idx] = BurstCoalescedLSU::load(aX_ptr + (i * 16 + vec_idx));
                    }
                    XVecPipe::write(ddr_read);
                }
            });
        }));

        kernel_events.push_back(q.submit([&](handler &h) {
            sycl::accessor aY(Y_buf, h, sycl::read_only);

            h.single_task<YLoader>([=]() [[intel::kernel_args_restrict]] {
                auto aY_ptr = aY.get_pointer();
                for (int i = 0; i < loop_iter; i++) {
                    float16 ddr_read;
#pragma unroll
                    for (int vec_idx = 0; vec_idx < 16; vec_idx++) {
                        ddr_read[vec_idx] = BurstCoalescedLSU::load(aY_ptr + (i * 16 + vec_idx));
                    }
                    YVecPipe::write(ddr_read);
                }
            });
        }));

        kernel_events.push_back(q.submit([&](handler &h) {
            sycl::accessor aZ(Z_buf, h, sycl::write_only, sycl::no_init);

            h.single_task<DotProduct>([=]() [[intel::kernel_args_restrict]] {
                float acc = 0.0f;
                for (int i = 0; i < loop_iter; i++) {
                    sycl::vec<float, 16> x_vec = XVecPipe::read();
                    sycl::vec<float, 16> y_vec = YVecPipe::read();
                    float partial_acc = dot_prod_16(x_vec, y_vec);
                    acc += partial_acc;
                }
                aZ[0] = acc;
            });
        }));

        for (unsigned int i = 0; i < kernel_events.size(); i++) {
            kernel_events.at(i).wait();
        }
    }

    double earliest_start_time =
        std::min_element(
            kernel_events.begin(), kernel_events.end(),
            [](const sycl::event &e1, const sycl::event &e2) {
                return e1.get_profiling_info<sycl::info::event_profiling::command_start>() <
                        e2.get_profiling_info<sycl::info::event_profiling::command_start>();
            })->get_profiling_info<sycl::info::event_profiling::command_start>();
    double latest_end_time =
        std::max_element(
            kernel_events.begin(), kernel_events.end(),
            [](const sycl::event &e1, const sycl::event &e2) {
                return e1.get_profiling_info<sycl::info::event_profiling::command_end>() <
                        e2.get_profiling_info<sycl::info::event_profiling::command_end>();
            })->get_profiling_info<sycl::info::event_profiling::command_end>();
    // Get time in ns
    double events_time = (latest_end_time - earliest_start_time);
    printf("  Time: %.5f ns\n", events_time);
    printf("  Throughput: %.2f GFLOPS\n", (double)2.0 * LVEC / events_time);

    // Check the results
    if (fabs(result - gold) > 0.005 * fabs(gold)) {
        std::cout << "result: " << result << ", gold: " << gold << std::endl;
        std::cout << "FAILED: result is incorrect!" << std::endl;
        return -1;
    }
    std::cout << "PASSED: result is correct!" << std::endl;
    return 0;
}
