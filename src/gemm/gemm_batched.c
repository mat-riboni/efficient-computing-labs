#include "kernels.h"

void gemm_batched_scalar(const float *a, const float *b, float *c,
                         size_t batch, size_t m, size_t n, size_t k) {
    /* TODO 7 — Batched scalar GEMM
       a stores batch independent row-major matrices A[batch,M,K]; b is one
       shared row-major weight matrix B[K,N]; c stores C[batch,M,N]. For each
       batch item, compute A[batch_i] * B using the ordinary GEMM dot product.
       Pseudocode: for t in batch, i in M, j in N: sum=0; for q in K:
       sum += a[(t*M+i)*K+q]*b[q*N+j]; c[(t*M+i)*N+j]=sum. */
    for (size_t bb = 0; bb < batch; ++bb) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t q = 0; q < k; ++q) {
                    sum += a[(bb * m * k) + (i * k) + q] * b[n * q + j];
                }
                c[(bb * m * n) + (i * n) + j] = sum;
            }
        }
    }
}

void gemm_batched_tiled(const float *a, const float *b, float *c,
                     size_t batch, size_t m, size_t n, size_t k, size_t tile) {
    /* TODO 8 — Batched tiled GEMM
       Arguments/layout are the same as gemm_batched_scalar; tile controls
       block widths. Tile each per-item output/reduction matrix as in
       gemm_tiled, but reuse the same b[K,N] for every batch item.
       Pseudocode: for t in batch, ii/jj output tiles, and qq reduction
       tiles: compute each output's full dot product into a local sum, then
       store it once in c[(t*M+i)*N+j]. */
    for (size_t bb = 0; bb < batch; ++bb) {
        for (size_t ii = 0; ii < m; ii += tile) {
            size_t i_end = m - ii < tile ? m : ii + tile;
            for (size_t jj = 0; jj < n; jj += tile) {
                size_t j_end = n - jj < tile ? n : jj + tile;
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t j = jj; j < j_end; ++j) {
                        float sum = 0.0f;
                        for (size_t kk = 0; kk < k; kk += tile) {
                            size_t q_end = k - kk < tile ? k : kk + tile;
                            for (size_t q = kk; q < q_end; ++q) {
                                sum += a[(bb * m * k) + (i * k) + q] *
                                       b[(q * n) + j];
                            }
                        }
                        c[(bb * m * n) + (i * n) + j] = sum;
                    }
                }
            }
         }
       }
    }

    // for (size_t ii = 0; ii < m; ii += tile) {
    //     size_t i_end = m - ii < tile ? m : ii + tile;
    //     for (size_t jj = 0; jj < n; jj += tile) {
    //         size_t j_end = n - jj < tile ? n : jj + tile;
    //         for (size_t bb = 0; bb < batch; ++bb) {
    //             for (size_t i = ii; i < i_end; ++i) {
    //                 for (size_t j = jj; j < j_end; ++j) {
    //                     float sum = 0.0f;
    //                     for (size_t kk = 0; kk < k; kk += tile) {
    //                         size_t q_end = k - kk < tile ? k : kk + tile;
    //                         for (size_t q = kk; q < q_end; ++q) {
    //                             sum += a[(bb * m * k) + (i * k) + q] *
    //                                    b[(q * n) + j];
    //                         }
    //                     }
    //                     c[(bb * m * n) + (i * n) + j] = sum;
    //                 }
    //             }
    //         }
    //     }
    // }
            
    //     }



void gemm_batched_omp_batch(const float *a, const float *b, float *c,
                            size_t batch, size_t m, size_t n, size_t k, size_t tile) {
    /* TODO 9 — OpenMP over batch
       Inputs/outputs have the same layout as above. Batch items write to
       disjoint C[t] regions, so parallelize the outer batch loop with one
       OpenMP parallel-for; keep each item's tiled GEMM serial. The shared
       B matrix is read-only. Pseudocode: parallel for t in batch: compute
       the complete tiled C[t] = A[t] * B. */
    #pragma omp parallel for schedule(static)
    for (size_t bb = 0; bb < batch; ++bb) {
        for (size_t ii = 0; ii < m; ii += tile) {
            size_t i_end = m - ii < tile ? m : ii + tile;
            for (size_t jj = 0; jj < n; jj += tile) {
                size_t j_end = n - jj < tile ? n : jj + tile;
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t j = jj; j < j_end; ++j) {
                        float sum = 0.0f;
                        for (size_t kk = 0; kk < k; kk += tile) {
                            size_t q_end = k - kk < tile ? k : kk + tile;
                            for (size_t q = kk; q < q_end; ++q) {
                                sum += a[(bb * m * k) + (i * k) + q] *
                                       b[(q * n) + j];
                            }
                        }
                        c[(bb * m * n) + (i * n) + j] = sum;
                    }
                }
            }
        }
    }
}

void gemm_batched_omp_tiles(const float *a, const float *b, float *c,
                            size_t batch, size_t m, size_t n, size_t k, size_t tile) {
    /* TODO 10 — OpenMP over output tiles
       Same batched/shared-weight layout. Parallelize independent (t,ii,jj)
       output tiles. Each worker owns its C tile and completes the full q/K
       reduction locally; do not split that reduction across workers, which
       would require synchronization and could race on partial sums.
       Pseudocode: parallel for each (t,ii,jj): for each output in that tile,
       compute its full sum over q=0..K-1 and store exactly once. */
    /* Each thread owns a distinct output tile and computes its full K reduction. */
    for (size_t bb = 0; bb < batch; ++bb) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t ii = 0; ii < m; ii += tile) {
            for (size_t jj = 0; jj < n; jj += tile) {
                size_t i_end = m - ii < tile ? m : ii + tile;
                size_t j_end = n - jj < tile ? n : jj + tile;
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t j = jj; j < j_end; ++j) {
                        float sum = 0.0f;
                        for (size_t kk = 0; kk < k; kk += tile) {
                            size_t q_end = k - kk < tile ? k : kk + tile;
                            for (size_t q = kk; q < q_end; ++q) {
                                sum += a[(bb * m * k) + (i * k) + q] *
                                       b[(q * n) + j];
                            }
                        }
                        c[(bb * m * n) + (i * n) + j] = sum;
                    }
                }
            }
        }
    }
}
