#include "dot_prod.h"

extern "C" float dot_prod(OCL_ADDRSP_GLOBAL float* A, OCL_ADDRSP_GLOBAL float* B, unsigned n) {
  float result = 0;
  for(unsigned i = 0; i < n; ++i) {
    result += A[i] * B[i]; 
  }

  return result;
}

#ifdef TEST_HLS_DOT_PROD

#include "HLS/hls.h"
#include <iostream>

#define N 6

using namespace std;

void print_array(float* A, unsigned n) {
  for(unsigned i = 0; i < n; ++i) {
    cout << A[i] <<  ' ';
  }
  cout << '\n';
}

component float dot_prod_comp(float* A, float* B, unsigned n) {
  return dot_prod(A, B, n);
}

int main() {
  float A[N];
  float B[N];

  // Initialize A and B
  for(unsigned i = 0; i < N; ++i) {
    A[i] = i;
    B[i] = i * N;
  }

  cout << "Vector A:\n";
  print_array(A, N);
  cout << "Vector B:\n";
  print_array(B, N);

  // Call dot_prod via the component and directly
  // Used to compare FPGA/simulation result with expected result
  float y = dot_prod(A, B, N);
  float y_fpga = dot_prod_comp(A, B, N);

  cout << "Dot product (expected): " << y << '\n';
  cout << "Dot product (result): " << y_fpga << '\n';

  // Compare results
  if(y_fpga == y) {
    cout << "PASSED\n";
  } else {
    cout << "FAILED\n";
  }
}

#endif // TEST_HLS_DOT_PROD
