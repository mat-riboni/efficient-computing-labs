#include "kernels.h"

float reduce_sum_scalar(const float *x, size_t n)
{
    /* TODO — Scalar sum reduction
       Argument: x points to n read-only FP32 elements. Return their sum; for
       n == 0, return 0.0f. A reduction combines many inputs into one result.
       Pseudocode:
         sum = 0
         for i = 0 .. n-1:
             sum += x[i]
         return sum
       Implement this scalar version first: the tests use it as the reference
       when checking the SIMD result. */
    float sum = 0.0f;

    for(size_t i = 0; i < n; ++i){
        sum += x[i];
    }
    return sum;
}
