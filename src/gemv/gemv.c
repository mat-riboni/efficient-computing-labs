#include "kernels.h"

void gemv_scalar(const float *a, const float *x, float *y, size_t m, size_t k) {
    /* TODO 3 — Scalar GEMV
       Arguments: a is row-major A[M,K], x has K elements, and y has M
       writable elements. Each result is the dot product of one matrix row
       with x: y[i] = sum(j=0..K-1) A[i,j] * x[j].
       Pseudocode:
         for i = 0 .. M-1:
             sum = 0
             for j = 0 .. K-1: sum += a[i*K+j] * x[j]
             y[i] = sum */
    /* The placeholder does not read these inputs/dimensions yet. The casts
       suppress unused-parameter warnings; remove them when implementing. */
    
    for (size_t i = 0; i < m; ++i) {

        float sum = 0;
        for (size_t j = 0; j < k; ++j){
            sum += a[i * k + j] * x[j];
        }
        
        y[i] = sum;
    }
}
