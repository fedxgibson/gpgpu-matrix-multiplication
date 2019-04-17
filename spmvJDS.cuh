#ifndef SPMV_CUH
#define SPMV_CUH

#include "constants.h"

float* spmvJDS_CPU(float* val, int* col_idx, int* col_start, int* perm_vector, float* x, int M, int N, int num_jdiags);

float* spmvJDSv0_hst(float* val, int* col_idx, int* col_start, int* perm_vector, float* x, int M, int N, int num_jdiags, int chunkSize);

float* spmvJDSv1_hst(float* val, int* col_idx, int* col_start, int* perm_vector, float* x, int M, int N, int num_jdiags, int chunkSize);

float* spmvJDSv2_hst(float* val, int* col_idx, int* col_start, int* perm_vector, float* x, int M, int N, int num_jdiags, int chunkSize);

float* spmvJDSv3_hst(float* val, int* col_idx, int* col_start, int* perm_vector, float* x, int M, int N, int num_jdiags, int chunkSize);

#endif