#include "kernels.h"
#include <arm_neon.h>

float reduce_sum_neon(const float *x, size_t n)
{
    /* TODO — NEON sum reduction
       A 128-bit NEON vector contains four FP32 lanes. Accumulate full groups
       of four with vaddq_f32 and vld1q_f32. Then horizontally add the lanes
       with vaddvq_f32 and add any remaining elements scalarly.
       Pseudocode:
         vector_sum = [0, 0, 0, 0]; i = 0
         while i + 4 <= n:
             vector_sum += load4(x + i)
             i += 4
         sum = horizontal_sum(vector_sum)
         while i < n:
             sum += x[i]; i += 1
         return sum
       Floating-point addition order differs from the scalar version, so
       compare results with a tolerance rather than exact equality. */

    float32x4_t vector_sum = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4){
        float32x4_t val = vld1q_f32(x + i);
        vector_sum = vaddq_f32(vector_sum, val);
    }

    float sum = vaddvq_f32(vector_sum);

    for (; i < n; ++i){
        sum += x[i];
    }
    return sum;
}
