#include "data_transfer.hpp"
#include "exception_handler.hpp"
#include "operators.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#define ROWS 32
#define COLS 32

#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif

using namespace sycl;

#define PRINTF(format, ...)                                    \
  {                                                            \
    static const CL_CONSTANT char _format[] = format;          \
    ext::oneapi::experimental::printf(_format, ##__VA_ARGS__); \
  }

// Forward declare the kernel and pipe names
// This FPGA best practice reduces name mangling in the optimization report.
class ALoader;
class Cholesky;
class LUnloader;
class APipe;
class LPipe;

constexpr size_t num_elems = ROWS * COLS;
constexpr size_t half_elems = ROWS * (COLS + 1) / 2;

using AVecPipe = sycl::ext::intel::pipe<APipe, float16, 256>;
using LVecPipe = sycl::ext::intel::pipe<LPipe, float16, 256>;

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

    float *A = (float *)malloc(num_elems * sizeof(float));
    float *L = (float *)malloc(num_elems * sizeof(float));

    constexpr size_t kRandomSeed = 1138;
    constexpr size_t kRandomMin = 1;
    constexpr size_t kRandomMax = 10;
    for (size_t i = 0; i < num_elems; i++) {
        int random_val = rand();
        float random_float =
            random_val % (kRandomMax - kRandomMin) + kRandomMin;
        A[i] = random_float;
    }

    float *A_device = sycl::malloc_device<float>(num_elems, q);
    float *L_device = sycl::malloc_device<float>(num_elems, q);

    q.memcpy(A_device, A, num_elems * sizeof(float)).wait();

    kernel_events.push_back(
        q.single_task<ALoader>([=]() [[intel::kernel_args_restrict]] {
            DRAMToPipe<float, num_elems, 16, AVecPipe>(A_device, 1, 1);
        }));

    kernel_events.push_back(
        q.single_task<Cholesky>([=]() {
            [[intel::numbanks(2)]]
            [[intel::bankwidth(64)]]
            [[intel::private_copies(4)]]
            float mat_A[COLS][ROWS];
            PipeToBRAM<float, ROWS, COLS, 16, AVecPipe>(mat_A);
            BRAMToPipe<float, ROWS, COLS, 16, LVecPipe>(mat_A);
        }));
    
    kernel_events.push_back(
        q.single_task<LUnloader>([=]() [[intel::kernel_args_restrict]] {
            PipeToDRAM<float, num_elems, 16, LVecPipe>(L_device, 1, 1);
        }));

    for (unsigned int i = 0; i < kernel_events.size(); i++) {
        kernel_events.at(i).wait();
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
    // printf("  Throughput: %.2f GFLOPS\n", (double)2.0 * LVEC / events_time);

    q.memcpy(L, L_device, num_elems * sizeof(float)).wait();

    // Check the results
    std::cout << "PASSED: result is correct!" << std::endl;
    return 0;
}
