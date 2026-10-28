#include "kernels.h"
#include <arm_neon.h>

void vector_add_neon(const float *a, const float *b, float *y, size_t n) {
    /* TODO 2 — NEON vector addition
       Arguments: a and b are read-only FP32 arrays of n elements; y is the
       writable FP32 result array. A NEON vector holds four FP32 values.
       Use vld1q_f32 to load, vaddq_f32 to add, and vst1q_f32 to store four
       elements per iteration. Then process any remaining elements scalarly.
       Pseudocode:
         i = 0
         while i + 4 <= n:
             va = load4(a+i); vb = load4(b+i); store4(y+i, va+vb)
             i += 4
         while i < n: y[i] = a[i] + b[i]; i += 1 */
    /* The placeholder does not read a or b yet. These casts mark the
       parameters as intentionally unused so the compiler does not warn. */

    size_t i = 0;

    for (; i < (size_t) n / 4; ++i) {
        float32x4_t a_parall = vld1q_f32(a + (i * 4));
        float32x4_t b_parall = vld1q_f32(b + (i * 4));
        float32x4_t sum = vaddq_f32(a_parall, b_parall);
        vst1q_f32(y + (i * 4), sum);
    }

    i = i * 4;

    for (; i < n; ++i){
        y[i] = a[i] + b[i];
    }
}
