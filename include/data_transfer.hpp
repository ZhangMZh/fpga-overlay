#ifndef __DATA_TRANSFERS_HPP__
#define __DATA_TRANSFERS_HPP__

#include <sycl/ext/intel/fpga_extensions.hpp>
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

template <typename T, int num_elems, int vec_len, typename VecPipe, typename LSU>
void DRAMToPipe(T *data_ptr, int batch_size, int repetitions) {
    sycl::device_ptr<T> data_ptr_device(data_ptr);
    constexpr int load_iter = num_elems / vec_len;
    for (int rep = 0; rep < repetitions; rep++) {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int li = 0; li < load_iter; li++) {
                sycl::vec<T, vec_len> vload;
#pragma unroll
                for (int vec_idx = 0; vec_idx < vec_len; vec_idx++) {
                    vload[vec_idx] = LSU::load(data_ptr_device + (li * vec_len + vec_idx));
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

template <typename T, int num_elems, int vec_len, typename VecPipe, typename LSU>
void PipeToDRAM(T *data_ptr, int batch_size, int repetitions) {
    sycl::device_ptr<T> data_ptr_device(data_ptr);
    constexpr int store_iter = num_elems / vec_len;
    for (int rep = 0; rep < repetitions; rep++) {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int si = 0; si < store_iter; si++) {
                sycl::vec<T, vec_len> vstore = VecPipe::read();
#pragma unroll
                for (int vec_idx = 0; vec_idx < vec_len; vec_idx++) {
                    LSU::store(data_ptr_device + (si * vec_len + vec_idx), vstore[vec_idx]);
                }
            }
        }
    }
}


template <typename T, int rows, int cols, int vec_len, typename VecPipe>
void PipeToBRAM(T bram_ptr[cols][rows]) {
    constexpr int store_iter_per_col = rows / vec_len;
    for (int col = 0; col < cols; col++) {
        for (int si = 0; si < store_iter_per_col; si++) {
            sycl::vec<T, vec_len> vstore = VecPipe::read();
#pragma unroll
            for (int usi = 0; usi < store_iter_per_col; usi++) {
#pragma unroll
                for (int vec_idx = 0; vec_idx < vec_len; vec_idx++) {
                    if (usi == si) {
                        bram_ptr[col][usi * vec_len + vec_idx] = vstore[vec_idx];
                    }
                    vstore[vec_idx] = sycl::ext::intel::fpga_reg(vstore[vec_idx]);
                }
            }
        }
    }
}

template <typename T, int rows, int cols, int vec_len, typename VecPipe>
void BRAMToPipe(T bram_ptr[cols][rows]) {
    constexpr int load_iter_per_col = rows / vec_len;
    for (int col = 0; col < cols; col++) {
        for (int li = 0; li < load_iter_per_col; li++) {
            sycl::vec<T, vec_len> vload;
#pragma unroll
            for (int uli = 0; uli < load_iter_per_col; uli++) {
#pragma unroll
                for (int vec_idx = 0; vec_idx < vec_len; vec_idx++) {
                    if (uli == li) {
                        vload[vec_idx] = bram_ptr[col][uli * vec_len + vec_idx];
                    } else {
                        vload[vec_idx] = sycl::ext::intel::fpga_reg(vload[vec_idx]);
                    }
                }
            }
            VecPipe::write(vload);
        }
    }
}

#endif /* __DATA_TRANSFERS_HPP__ */
