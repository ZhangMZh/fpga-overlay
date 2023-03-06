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

#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

// !!! IMPORTANT !!!
// currently changing the matrix dimensions requires a recompile of both the FPGA code and the HOST code

// Matrix dimensions 
#ifdef REDUCE_DATA
#define SCALING_FACTOR 4
#else
#define SCALING_FACTOR 32
#endif

#define HA (4 * MAT_A_BLOCK_HEIGHT)             // Matrix A height
#define WA (SCALING_FACTOR * MAT_A_BLOCK_WIDTH) // Matrix A width

#define HB WA                                   // Matrix B height
#define WB (4 * MAT_B_BLOCK_WIDTH)              // Matrix B width

#define HC HA                                   // Matrix C height
#define WC WB                                   // Matrix C width 


// A+B+C matrices = 1.47 GB (with scaling factor 24)
// S10 EA board DDR memory = 2 GB


#define MAT_A_NUM_BLOCKS_IN_ROW             (WA / MAT_A_BLOCK_WIDTH)
#define MAT_A_NUM_BLOCKS_IN_COL             (HA / MAT_A_BLOCK_HEIGHT)
#define MAT_A_NUM_VECTORS_IN_ROW_OF_BLOCKS  (MAT_A_NUM_BLOCKS_IN_ROW * MAT_A_BLOCK_NUM_VECTORS)

#define MAT_B_NUM_BLOCKS_IN_ROW             (WB / MAT_B_BLOCK_WIDTH)
#define MAT_B_NUM_BLOCKS_IN_COL             (HB / MAT_B_BLOCK_HEIGHT)
#define MAT_B_NUM_VECTORS_IN_COL_OF_BLOCKS  (MAT_B_NUM_BLOCKS_IN_COL * MAT_B_BLOCK_NUM_VECTORS)
#define MAT_B_NUM_VECTORS_IN_MATRIX         (MAT_B_NUM_VECTORS_IN_COL_OF_BLOCKS * MAT_B_NUM_BLOCKS_IN_ROW)

#endif // _MATRIXMUL_H_

