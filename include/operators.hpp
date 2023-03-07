#ifndef __OPERATORS_HPP__
#define __OPERATORS_HPP__
#include <sycl/sycl.hpp>

SYCL_EXTERNAL float dot_prod_16(sycl::float16 x, sycl::float16 y);
// template <typename T, int N>
// SYCL_EXTERNAL T dot_prod(sycl::vec<T, N> x, sycl::vec<T, N> y);

#endif //__OPERATORS_HPP__
