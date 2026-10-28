#define _POSIX_C_SOURCE 200809L
#include "kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { WARMUPS = 20, SAMPLES = 100 };
typedef void (*kernel_fn)(const float *, const float *, float *, size_t);
static const float *input_a, *input_b;
static float *output;
static size_t length;

static double now_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int compare_double(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void report(const char *name, const double *values)
{
    double sorted[SAMPLES], mean = 0.0, variance = 0.0;
    for (int i = 0; i < SAMPLES; ++i) {
        sorted[i] = values[i];
        mean += values[i] / SAMPLES;
    }
    qsort(sorted, SAMPLES, sizeof(sorted[0]), compare_double);
    for (int i = 0; i < SAMPLES; ++i) {
        const double delta = values[i] - mean;
        variance += delta * delta / SAMPLES;
    }
    printf("%-12s %10.3f %10.3f %10.3f %10.3f\n", name,
           (sorted[49] + sorted[50]) * 0.5, sqrt(variance), sorted[0], sorted[99]);
}

static void run_kernel(const char *name, kernel_fn fn)
{
    double samples[SAMPLES];
    for (int i = 0; i < WARMUPS; ++i)
        fn(input_a, input_b, output, length);
    for (size_t i = 0; i < length; ++i) {
        if (!isfinite(output[i]) || output[i] != input_a[i] + input_b[i]) {
            fprintf(stderr, "%s correctness check failed at element %zu\n", name, i);
            exit(EXIT_FAILURE);
        }
    }
    for (int i = 0; i < SAMPLES; ++i) {
        const double start = now_seconds();
        fn(input_a, input_b, output, length);
        samples[i] = (now_seconds() - start) * 1e6;
    }
    report(name, samples);
}

int main(void)
{
    length = 1045876;
    float *a = malloc(length * sizeof(*a));
    float *b = malloc(length * sizeof(*b));
    output = malloc(length * sizeof(*output));
    if (a == NULL || b == NULL || output == NULL) {
        fprintf(stderr, "Could not allocate vector-add benchmark arrays\n");
        free(a); free(b); free(output);
        return 2;
    }
    input_a = a; input_b = b;
    for (size_t i = 0; i < length; ++i) {
        a[i] = (float)(i % 17);
        b[i] = (float)(i % 11);
    }
    puts("Vector addition: y[i] = a[i] + b[i]");
    printf("Warmups: %d, timed samples: %d (microseconds)\n", WARMUPS, SAMPLES);
    puts("Method             median       std        min        max");
    run_kernel("Scalar", vector_add_scalar);
    run_kernel("NEON", vector_add_neon);
    free(a); free(b); free(output);
    return 0;
}
