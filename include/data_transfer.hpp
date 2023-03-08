#ifndef __MEMORY_TRANSFERS_HPP__
#define __MEMORY_TRANSFERS_HPP__

#include <sycl/sycl.hpp>

template <typename T, int num_elems, int vec_len, typename VecPipe>
void DRAMToPipe(T *data_ptr, int batch_size, int repetitions) {
    sycl::device_ptr<T> data_ptr_device(data_ptr);
    constexpr int load_iter = num_elems / vec_len;
    for (int rep = 0; rep < repetitions; rep++) {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int li = 0; li < load_iter; li++) {
                sycl::vec<T, vec_len> vload;
#pragma unroll
                for (int vec_idx = 0; vec_idx < vec_len; vec_idx++) {
                    vload[vec_idx] = data_ptr_device[li * vec_len + vec_idx];
                }
                VecPipe::write(vload);
            }
        }
    }
}

template <typename T, int num_elems, int vec_len, typename VecPipe>
void PipeToDRAM(T *data_ptr, int batch_size, int repetitions) {
    sycl::device_ptr<T> data_ptr_device(data_ptr);
    constexpr int store_iter = num_elems / vec_len;
    for (int rep = 0; rep < repetitions; rep++) {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int si = 0; si < store_iter; si++) {
                sycl::vec<T, vec_len> vstore = VecPipe::read();
#pragma unroll
                for (int vec_idx = 0; vec_idx < vec_len; vec_idx++) {
                    data_ptr_device[si * vec_len + vec_idx] = vstore[vec_idx];
                }
            }
        }
    }
}

template <typename T, int num_elems, int vec_len, typename VecPipe>
void PipeToBRAM(T *data_ptr, int batch_size, int repetitions) {
    sycl::device_ptr<T> data_ptr_device(data_ptr);
    constexpr int store_iter = num_elems / vec_len;
    for (int rep = 0; rep < repetitions; rep++) {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int si = 0; si < store_iter; si++) {
                sycl::vec<T, vec_len> vstore = VecPipe::read();
#pragma unroll
                for (int vec_idx = 0; vec_idx < vec_len; vec_idx++) {
                    data_ptr_device[si * vec_len + vec_idx] = vstore[vec_idx];
                }
            }
        }
    }
}

#endif /* __MEMORY_TRANSFERS_HPP__ */
