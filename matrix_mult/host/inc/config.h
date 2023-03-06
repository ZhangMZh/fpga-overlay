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

#ifndef _PE_SYSTOLIC_ARRAY_H_
#define _PE_SYSTOLIC_ARRAY_H_

// This is important but it is not enforced:
// PE_ROWS + PE_COLS <= ROWS_INTERLEAVED

// design space exploration of three vector sizes: float4, float8 and float16
#ifndef DOT_PROD_VECTOR_SIZE
   #define DOT_PROD_VECTOR_SIZE     8
#endif

#ifndef FORCE_DOT_4
   #define FORCE_DOT_4              0
#endif

#ifndef PE_ROWS
   #define PE_ROWS                  2
#endif
#ifndef PE_COLS
   #define PE_COLS                  2
#endif

#ifndef ROWS_INTERLEAVED
#ifdef REDUCE_DATA
   #define ROWS_INTERLEAVED         8
#else       
   #define ROWS_INTERLEAVED         32
#endif
#endif

#ifndef COLUMNS_INTERLEAVED
#ifdef REDUCE_DATA
   #define COLUMNS_INTERLEAVED      8
#else
   #define COLUMNS_INTERLEAVED      32
#endif
#endif

#define ACCUM_SHIFT_REG_SIZE        (ROWS_INTERLEAVED * COLUMNS_INTERLEAVED)
#define C_OUT_SHIFT_REG_SIZE        ACCUM_SHIFT_REG_SIZE
    
#define MAT_A_BLOCK_WIDTH           (16 * DOT_PROD_VECTOR_SIZE)
#define MAT_A_BLOCK_HEIGHT          (ROWS_INTERLEAVED   * PE_ROWS)
#define MAT_A_BLOCK_SIZE            (MAT_A_BLOCK_HEIGHT * MAT_A_BLOCK_WIDTH)
#define MAT_A_BLOCK_NUM_VECTORS     (MAT_A_BLOCK_SIZE   / DOT_PROD_VECTOR_SIZE)

#define MAT_B_BLOCK_HEIGHT          MAT_A_BLOCK_WIDTH
#define MAT_B_BLOCK_WIDTH           (COLUMNS_INTERLEAVED * PE_COLS)
#define MAT_B_BLOCK_SIZE            (MAT_B_BLOCK_HEIGHT  * MAT_B_BLOCK_WIDTH)
#define MAT_B_BLOCK_NUM_VECTORS     (MAT_B_BLOCK_SIZE    / DOT_PROD_VECTOR_SIZE)
    
#define MAT_C_BLOCK_HEIGHT          MAT_A_BLOCK_HEIGHT
#define MAT_C_BLOCK_WIDTH           MAT_B_BLOCK_WIDTH

#define VECTOR_FLOAT4_ZERO          (float4)(0.0f, 0.0f, 0.0f, 0.0f)
#define VECTOR_FLOAT8_ZERO          (float8)(VECTOR_FLOAT4_ZERO,VECTOR_FLOAT4_ZERO)
#define VECTOR_FLOAT16_ZERO         (float16)(VECTOR_FLOAT8_ZERO,VECTOR_FLOAT8_ZERO)

#define VEC                         DOT_PROD_VECTOR_SIZE
#define ROWS                        PE_ROWS
#define COLS                        PE_COLS
#define COLS_INTERLEAVED            COLUMNS_INTERLEAVED

#define ROW_VECS                    (MAT_A_BLOCK_WIDTH / VEC)
#define ROW_VECS_MASK               (ROW_VECS - 1)

#define SWAP_RANGE                  (ROWS_INTERLEAVED * COLS_INTERLEAVED * ROW_VECS)
#define SWAP_RANGE_MASK             (SWAP_RANGE - 1)

#define RANGE                       (2 * SWAP_RANGE)
#define RANGE_MASK                  (RANGE - 1)

#ifndef HOST
    #if DOT_PROD_VECTOR_SIZE==4
        typedef float4 vec_float_t;
        #define VECTOR_ZERO         VECTOR_FLOAT4_ZERO
    #elif DOT_PROD_VECTOR_SIZE==8
        typedef float8 vec_float_t;
        #define VECTOR_ZERO         VECTOR_FLOAT8_ZERO
    #elif DOT_PROD_VECTOR_SIZE==16
        typedef float16 vec_float_t;
        #define VECTOR_ZERO         VECTOR_FLOAT16_ZERO
    #else
        #error Unsupported DOT_PROD_VECTOR_SIZE
#endif

struct vec_float_t_bool {
    vec_float_t data;
    bool  c;  // indicates a new row/column pair
};

#ifndef LVEC
   #define LVEC 1
#endif

struct nvec_float_t {
    vec_float_t data[LVEC];
};

struct nvec_float_t_bool {
    vec_float_t data[LVEC];
    bool  c;  // indicates a new row/column pair
};

struct cols_floats { 
    float drain_data[PE_COLS];
};

// The number of rows rounded up to the next power of 2
#if ROWS <= 1
    #define BANKROWS 1
#elif ROWS <= 2
    #define BANKROWS 2
#elif ROWS <= 4
    #define BANKROWS 4
#elif ROWS <= 8
    #define BANKROWS 8
#elif ROWS <= 16
    #define BANKROWS 16
#elif ROWS <= 32
    #define BANKROWS 32
#elif ROWS <= 64
    #define BANKROWS 64
#elif ROWS <= 128
    #define BANKROWS 128
#elif ROWS <= 256
    #define BANKROWS 256
#else
    #error "ROWS too large, BANKROWS cannot be defined"
#endif

// The number of columns rounded up to the next power of 2
#if COLS <= 1
    #define BANKCOLS 1
#elif COLS <= 2
    #define BANKCOLS 2
#elif COLS <= 4
    #define BANKCOLS 4
#elif COLS <= 8
    #define BANKCOLS 8
#elif COLS <= 16
    #define BANKCOLS 16
#elif COLS <= 32
    #define BANKCOLS 32
#elif COLS <= 64
    #define BANKCOLS 64
#elif COLS <= 128
    #define BANKCOLS 128
#elif COLS <= 256
    #define BANKCOLS 256
#else
    #error "COLS too large, BANKCOLS cannot be defined"
#endif


#endif

#endif // _PE_SYSTOLIC_ARRAY_H_

