#include "sparse_gemm.h"
#include <stdio.h>
#include <arm_neon.h>


int gemm_sparse_openmp_tiled_neon(const float *a, const sparse_weights *w,
                                  float *c, size_t m)
{   

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < m; ++i){
        for (size_t j = 0; j < w->n; ++j){
            c[i * w->n + j] = 0.0f;
        }
    }
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t ii = 0; ii < m; ii += 4){
        for (size_t g = 0; g < w->n / 4; ++g) {
            for (size_t p = w->block_col_ptr[g]; p < w->block_col_ptr[g + 1]; ++p) {
                size_t q = w->block_row_idx[p];
                size_t k0 = 4 * q;
                for (size_t i = ii; i < ii + 4; ++i){
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (size_t r = 0; r < 4; ++r) {
                        float av = a[i * w->k + k0 + r];
                        acc += av * vld1q_f32(w->block_values + (16 * p + 4 * r));
                    }
                    vst1q_f32(c + (i * w->n + 4 * g), acc + vld1q_f32(c + (i * w->n + 4 * g))); 
                }
            }
            
        }
    }
    return SPARSE_OK;
}
