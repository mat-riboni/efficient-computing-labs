#include "sparse_gemm.h"
#include <stdio.h>
#include <arm_neon.h>

int gemm_sparse_neon(const float *a, const sparse_weights *w, float *c, size_t m)
{
    if (!a || !w || !c)
        return SPARSE_ERROR;
    for (size_t i = 0; i < m; ++i)
        for (size_t g = 0; g < w->n / 4; ++g) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (size_t p = w->block_col_ptr[g]; p < w->block_col_ptr[g + 1]; ++p) {
                size_t k0 = 4 * (size_t)w->block_row_idx[p];
                for (size_t r = 0; r < 4; ++r) {
                    float av = a[i * w->k + k0 + r];
                    acc += av * vld1q_f32(w->block_values + (16 * p + 4 * r));
                }
            }
            vst1q_f32(c +(i * w->n + 4 * g), acc); 
        }
    return SPARSE_OK;
}
