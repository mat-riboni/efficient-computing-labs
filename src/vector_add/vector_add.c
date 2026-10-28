#include "kernels.h"

void vector_add_scalar(const float *a, const float *b, float *y, size_t n) {
    /* TODO 1 — Scalar vector addition
       Arguments: a and b are read-only FP32 arrays of n elements; y is the
       writable FP32 result array of n elements. Compute y[i] = a[i] + b[i].
       Pseudocode:
         for i = 0 .. n-1:
             y[i] = a[i] + b[i]
       Each output element is independent. */
    /* The placeholder does not read a or b yet. These casts mark the
       parameters as intentionally unused so the compiler does not warn. */
    
    for (size_t i = 0; i < n; ++i) {
        y[i] = a[i] + b[i];
    }
}
