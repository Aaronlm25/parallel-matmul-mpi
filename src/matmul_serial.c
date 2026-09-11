/*
 * matmul_serial.c
 * ----------------
 * Sequential baseline for dense N x N matrix multiplication: C = A * B
 *
 * This program exists purely as the reference point used to compute
 * speedup and parallel efficiency for the MPI version (matmul_mpi.c).
 * It uses the same data generation, memory layout and multiplication
 * order as the parallel version so the comparison is fair.
 *
 * Usage:
 *   ./matmul_serial <N> [seed]
 *
 * Prints the wall-clock time (seconds) taken by the multiplication only
 * (matrix generation/allocation is excluded from the timed region).
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Allocate a contiguous N*N block so rows are cache-friendly and the
 * whole matrix can be sent over MPI in a single message when needed. */
static double *alloc_matrix(int n) {
    double *m = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!m) {
        fprintf(stderr, "Error: could not allocate a %dx%d matrix\n", n, n);
        exit(EXIT_FAILURE);
    }
    return m;
}

static void fill_random(double *m, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n * n; i++) {
        m[i] = (double)(rand() % 100) / 10.0; /* values in [0.0, 9.9] */
    }
}

/* Naive triple loop, i-k-j order for better cache locality on the B/C rows. */
static void matmul(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
        }
        for (int k = 0; k < n; k++) {
            double a_ik = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <N> [seed]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    unsigned int seed = (argc > 2) ? (unsigned int)atoi(argv[2]) : 42u;

    if (n <= 0) {
        fprintf(stderr, "N must be a positive integer\n");
        return EXIT_FAILURE;
    }

    double *A = alloc_matrix(n);
    double *B = alloc_matrix(n);
    double *C = alloc_matrix(n);

    /* Same seed for A and B across serial/MPI runs so results are
     * numerically comparable (used by matmul_mpi.c's --verify mode). */
    fill_random(A, n, seed);
    fill_random(B, n, seed + 1);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    matmul(A, B, C, n);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    /* Machine-parseable line first (used by scripts/benchmark.sh),
     * then a human-readable summary. */
    printf("RESULT,serial,1,%d,%.6f\n", n, elapsed);
    printf("N=%d  processes=1 (serial)  time=%.6fs\n", n, elapsed);

    /* Checksum so we can sanity-check correctness without printing
     * the whole matrix. */
    double checksum = 0.0;
    for (int i = 0; i < n * n; i++) checksum += C[i];
    printf("checksum=%.6f\n", checksum);

    free(A);
    free(B);
    free(C);
    return EXIT_SUCCESS;
}
