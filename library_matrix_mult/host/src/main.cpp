// Copyright (C) 2013-2019 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

#define N1 4
#define N2 8
#define N3 6

#define epsilon 1.0e-4f

using namespace aocl_utils;

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue cq;
cl_program program;
cl_kernel kernel;
cl_int status;

// Randomly generate a floating-point number between -100 and 100.
float random_float() {
  return ((float) rand() / (float) RAND_MAX) * 20.0f - 10.0f;
}

void random_array(float* A, unsigned size) {
  for(unsigned i = 0; i < size; ++i) {
    A[i] = random_float();
  }
}

void print_matrix(const float* M, unsigned n1, unsigned n2) {
  for(unsigned i = 0; i < n1; ++i) {
    for(unsigned j = 0; j < n2; ++j) {
      printf("%f ", M[i*n2 + j]);
    }
    printf("\n");
  }
}
    

void golden_matrix_mult(const float* A, const float* B, float* C, unsigned n1, unsigned n2, unsigned n3) {
  for(unsigned i = 0; i < n1; ++i) {
    for(unsigned j = 0; j < n3; ++j) {
      float sum = 0;
      for(unsigned k = 0; k < n2; ++k) {
        sum += A[i*n2 + k] * B[k*n3 + j];
      }
      C[i*n3 + j] = sum;
    }
  }
}

// Device memory for the matrices
cl_mem d_A;
cl_mem d_B;
cl_mem d_C;

// Host memory for the matrices
float* h_A;
float* h_B;
float* h_C;
float* h_C_golden;

unsigned n1 = N1;
unsigned n2 = N2;
unsigned n3 = N3;

void cleanup() {
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(cq);
  clReleaseContext(context);
  clReleaseMemObject(d_A);
  clReleaseMemObject(d_B);
  clReleaseMemObject(d_C);
  free(h_A);
  free(h_B);
  free(h_C);
}

int main(int argc, char**argv)
{
  Options options(argc, argv);

  // Optional arguments to specify the matrix sizes
  if(options.has("n1")) {
    n1 = options.get<unsigned>("n1");
  }

  if(options.has("n2")) {
    n1 = options.get<unsigned>("n2");
  }

  if(options.has("n3")) {
    n1 = options.get<unsigned>("n3");
  }

  cl_int status;
  
  printf("Initializing OpenCL\n");
  
  printf("Get platform and device IDs...\n");
  status = clGetPlatformIDs(1, &platform, 0);
  checkError(status, "Failed to get platform IDs");
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device, 0);
  checkError(status, "Failed to get device IDs");
 
  // Create the context
  context = clCreateContext(0, 1, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");
  
  // Set seed for rand()
  srand(42);
  
  // Allocate host memory
  unsigned size_A = n1 * n2;
  unsigned mem_size_A = sizeof(float) * size_A;
  h_A = (float*) malloc(mem_size_A);
  random_array(h_A, size_A);
  printf("Matrix A:\n");
  print_matrix(h_A, n1, n2);
  
  unsigned size_B = n2 * n3;
  unsigned mem_size_B = sizeof(float) * size_B;
  h_B = (float*) malloc(mem_size_B);
  random_array(h_B, size_B);
  printf("Matrix B:\n");
  print_matrix(h_B, n2, n3);
  
  unsigned size_C = n1 * n3;
  unsigned mem_size_C = sizeof(float) * size_C;
  h_C = (float*) malloc(mem_size_C);

  h_C_golden = (float*) malloc(mem_size_C);
  
  printf("Create command queue...\n");
  cq = clCreateCommandQueue(context, device, 0, &status);
  checkError(status, "Failed to create command queue");
  
  printf("Load kernel...\n");
  std::string binary_file = getBoardBinaryFile("bin/kernel", device);
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);
  
  printf("Build kernel program...\n");
  status = clBuildProgram(program, 1, &device, "", 0, 0);
  checkError(status, "Failed to build kernel");
  
  printf("Create kernel...\n");
  kernel = clCreateKernel(program,"func",&status);
  checkError(status, "Failed to create kernel");
  
  printf("Create device buffers...\n");
  d_A = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_A, NULL, &status);
  d_B = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_B, NULL, &status);
  d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_C, NULL, &status);
  checkError(status, "Failed to allocate memory for device");
  
  printf("Set kernel arguments...\n");
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
  status = clSetKernelArg(kernel, 3, sizeof(unsigned), &n1);
  status = clSetKernelArg(kernel, 4, sizeof(unsigned), &n2);
  status = clSetKernelArg(kernel, 5, sizeof(unsigned), &n3);
  checkError(status, "Failed to set kernel arguments");
  
  printf("Write to input buffers...\n");
  status = clEnqueueWriteBuffer(cq, d_A, CL_TRUE, 0, mem_size_A, h_A, 0, NULL, NULL);
  status = clEnqueueWriteBuffer(cq, d_B, CL_TRUE, 0, mem_size_B, h_B, 0, NULL, NULL);
  checkError(status, "Failed to enqueue buffer write");
  
  printf("Run kernel...\n");
  status = clEnqueueTask(cq, kernel, 0, NULL, NULL);
  checkError(status, "Failed to enqueue kernel task");
  
  printf("Read result buffer...\n");
  status = clEnqueueReadBuffer(cq, d_C, CL_TRUE, 0, mem_size_C, h_C, 0, NULL, NULL);
  checkError(status, "Failed to enqueue buffer read");
  
  // Block until all commands finish
  clFinish(cq);
  
  golden_matrix_mult(h_A, h_B, h_C_golden, n1, n2, n3);

  printf("Matrix C (expected):\n");
  print_matrix(h_C_golden, n1, n3);

  printf("Matrix C (result):\n");
  print_matrix(h_C, n1, n3);

  bool passed = true;
  for(unsigned i = 0; i < n1; ++i) {
    for(unsigned j = 0; j < n3; ++j) {
      float diff = fabs(h_C[i*n3 + j] - h_C_golden[i*n3 + j]);
      if(diff > epsilon) {
        passed = false;
        printf("Diff: %f\n", diff);
      }
    }
  }
  
  if(passed) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
  }

  cleanup();
  
  return 0;
}
