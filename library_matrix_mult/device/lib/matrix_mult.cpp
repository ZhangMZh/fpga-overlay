#include "dot_prod.h"

// In-place transpose using "follow-the-cycles" algorthm
void transpose(OCL_ADDRSP_GLOBAL float* M, unsigned n1, unsigned n2) {
  for(unsigned start = 0; start <= n1*n2; ++start) {
    unsigned next = start;
    unsigned i = 0;

    // Compute length i of the permutation cycle. Eiter next ends up back at start, or
    // it ends up touching a position we've already handled.
    do {
      ++i;
      next = (next % n1)*n2 + next/n1;
    } while(next > start);

    // Only operate on cycles of size > 1
    if(next == start && i != 1) {
      const float tmp = M[start];

      // Shift the elements in the permutation cycle.
      do {
        i = (next % n1)*n2 + next/n1;
        M[next] = (i == start) ? tmp : M[i];
        next = i;
      } while(next > start);
    }
  }
}

// Matrix multiply C = AxB where
// A is n1 by n2
// B is n2 by n3 and
// C is n1 by n3
extern "C" void matrix_mult(OCL_ADDRSP_GLOBAL float* A, OCL_ADDRSP_GLOBAL float* B,
                            OCL_ADDRSP_GLOBAL float* C, unsigned n1, unsigned n2, unsigned n3) {

  // Make B column-major order for call to dot product
  transpose(B, n2, n3);

  for(unsigned i = 0; i < n1; ++i) {
    for(unsigned j = 0; j < n3; ++j) {
      C[i*n3 + j] = dot_prod(&A[i*n2], &B[j*n2], n2);
    }
  }

  // Reset B
  transpose(B, n3, n2);
}

#ifdef TEST_HLS_MATRIX_MULT

#include "HLS/hls.h"
#include <iostream>

#define N1 4
#define N2 8
#define N3 6

using namespace std;

void print_matrix(float* M, unsigned n1, unsigned n2) {
  for(unsigned i = 0; i < n1; ++i) {
    for(unsigned j = 0; j < n2; ++j) {
      cout << M[i*n2 + j] << ' ';
    }
    cout << '\n';
  }
}

component void matrix_mult_comp(float* A, float* B, float* C, unsigned n1, unsigned n2, unsigned n3) {
  matrix_mult(A, B, C, n1, n2, n3);
}

int main() {
  float A[N1*N2];
  float B[N2*N3];
  float C[N1*N3];
  float C_fpga[N1*N3];

  // Initialize A and B
  for(unsigned i = 0; i < N1; ++i) {
    for(unsigned j = 0; j < N2; ++j) {
      A[i*N2 + j] = i + j;
    }
  }

  for(unsigned i = 0; i < N2; ++i) {
    for(unsigned j = 0; j < N3; ++j) {
      B[i*N3 + j] = (i + 1)*(j + 1);
    }
  }
  // Call matrix_mult via the component and directly
  // Used to compare FPGA/simulation result with expected result
  cout << "Matrix A:\n";
  print_matrix(A, N1, N2);
  cout << "Matrix B:\n";
  print_matrix(B, N2, N3);

  matrix_mult_comp(A, B, C, N1, N2, N3);
  cout << "Matrix C (expected):\n";
  print_matrix(C, N1, N3);

  matrix_mult_comp(A, B, C_fpga, N1, N2, N3);
  cout << "Matrix C (result):\n";
  print_matrix(C_fpga, N1, N3);

  // Compare results
  bool passed = true;
  for(unsigned i=0; i<N1; ++i) {
    for(unsigned j=0; j<N3; ++j) {
      if(C_fpga[i*N3 + j] != C[i*N3 + j]) {
        passed = false;
      }
    }
  }

  if(passed) {
    cout << "PASSED\n";
  } else {
    cout << "FAILED\n";
  }

  return 0;
}

#endif // TEST_HLS_MATRIX_MULT
