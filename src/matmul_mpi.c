/*
 * matmul_mpi.c
 * ------------
 * Parallel dense N x N matrix multiplication: C = A * B, using MPI.
 *
 * Strategy (row-wise decomposition):
 *   1. Rank 0 generates A and B.
 *   2. B is broadcast in full to every rank (every rank needs all of B
 *      to compute any row of C).
 *   3. The rows of A are distributed across ranks with MPI_Scatterv,
 *      using explicit counts/displacements so N does NOT need to be
 *      evenly divisible by the number of processes (the first
 *      N % size ranks simply get one extra row each). This is the
 *      "efficient memory handling" part referenced in the README:
 *      no rank ever holds more of A than the rows it actually needs,
 *      and no process is left idle or overloaded due to bad rounding.
 *   4. Each rank computes its own block of rows of C locally.
 *   5. The partial results are collected back on rank 0 with
 *      MPI_Gatherv.
 *
 * Complexity (see README.md for the full derivation):
 *   Serial:    O(N^3)
 *   Per rank:  O(N^3 / p) computation + O(N^2) communication
 *              (dominated by the B broadcast)
 *
 * Usage:
 *   mpirun -n <P> ./matmul_mpi <N> [seed] [--verify]
 *
 *   --verify   Rank 0 additionally computes the product serially and
 *              compares it against the gathered parallel result,
 *              printing PASS/FAIL. Only intended for small N during
 *              development/testing, not for benchmarking runs.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double *alloc_matrix(int n) {
    double *m = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!m) {
        fprintf(stderr, "Error: could not allocate a %dx%d matrix\n", n, n);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return m;
}

static void fill_random(double *m, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n * n; i++) {
        m[i] = (double)(rand() % 100) / 10.0;
    }
}

/* Multiplies `rows` rows of A (starting at row 0 of the local buffer)
 * by the full B matrix, writing into `rows` rows of C. */
static void matmul_rows(const double *A_local, const double *B, double *C_local,
                         int rows, int n) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < n; j++) {
            C_local[i * n + j] = 0.0;
        }
        for (int k = 0; k < n; k++) {
            double a_ik = A_local[i * n + k];
            for (int j = 0; j < n; j++) {
                C_local[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

/* Computes how many rows each rank owns and at what row offset,
 * spreading the N % size remainder over the first ranks so no rank
 * ever gets more than one extra row. */
static void compute_row_distribution(int n, int size, int *rows_per_rank,
                                      int *row_offset) {
    int base = n / size;
    int rem = n % size;
    int offset = 0;
    for (int r = 0; r < size; r++) {
        rows_per_rank[r] = base + (r < rem ? 1 : 0);
        row_offset[r] = offset;
        offset += rows_per_rank[r];
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "Usage: %s <N> [seed] [--verify]\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    unsigned int seed = 42u;
    int verify = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--verify") == 0) {
            verify = 1;
        } else {
            seed = (unsigned int)atoi(argv[i]);
        }
    }

    if (n <= 0) {
        if (rank == 0) fprintf(stderr, "N must be a positive integer\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (n < size && rank == 0) {
        fprintf(stderr, "Warning: N (%d) < number of processes (%d); "
                        "some processes will get 0 rows.\n", n, size);
    }

    int *rows_per_rank = (int *)malloc(size * sizeof(int));
    int *row_offset = (int *)malloc(size * sizeof(int));
    compute_row_distribution(n, size, rows_per_rank, row_offset);

    /* counts/displacements in *elements* (rows * n), not rows, since
     * that is what Scatterv/Gatherv operate on. */
    int *send_counts = (int *)malloc(size * sizeof(int));
    int *send_displs = (int *)malloc(size * sizeof(int));
    for (int r = 0; r < size; r++) {
        send_counts[r] = rows_per_rank[r] * n;
        send_displs[r] = row_offset[r] * n;
    }

    double *A = NULL, *B = alloc_matrix(n), *C = NULL;
    double *A_serial_copy = NULL; /* only used by rank 0 in --verify mode */

    if (rank == 0) {
        A = alloc_matrix(n);
        C = alloc_matrix(n);
        fill_random(A, n, seed);
        fill_random(B, n, seed + 1);
        if (verify) {
            A_serial_copy = alloc_matrix(n);
            memcpy(A_serial_copy, A, (size_t)n * n * sizeof(double));
        }
    }

    int my_rows = rows_per_rank[rank];
    double *A_local = (double *)malloc((size_t)my_rows * n * sizeof(double));
    double *C_local = (double *)malloc((size_t)my_rows * n * sizeof(double));
    if ((my_rows > 0) && (!A_local || !C_local)) {
        fprintf(stderr, "Rank %d: allocation failure\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* Every rank needs the whole of B to multiply its rows of A. */
    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Distribute A's rows: each rank gets exactly rows_per_rank[rank]. */
    MPI_Scatterv(A, send_counts, send_displs, MPI_DOUBLE,
                 A_local, my_rows * n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    matmul_rows(A_local, B, C_local, my_rows, n);

    /* Collect every rank's rows of C back into the full matrix on rank 0. */
    MPI_Gatherv(C_local, my_rows * n, MPI_DOUBLE,
                C, send_counts, send_displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    if (rank == 0) {
        printf("RESULT,mpi,%d,%d,%.6f\n", size, n, elapsed);
        printf("N=%d  processes=%d  time=%.6fs\n", n, size, elapsed);

        double checksum = 0.0;
        for (int i = 0; i < n * n; i++) checksum += C[i];
        printf("checksum=%.6f\n", checksum);

        if (verify) {
            double *C_ref = alloc_matrix(n);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) C_ref[i * n + j] = 0.0;
                for (int k = 0; k < n; k++) {
                    double a_ik = A_serial_copy[i * n + k];
                    for (int j = 0; j < n; j++) C_ref[i * n + j] += a_ik * B[k * n + j];
                }
            }
            double max_diff = 0.0;
            for (int i = 0; i < n * n; i++) {
                double d = fabs(C_ref[i] - C[i]);
                if (d > max_diff) max_diff = d;
            }
            printf("verify: max_abs_diff=%.3e -> %s\n", max_diff,
                   (max_diff < 1e-6) ? "PASS" : "FAIL");
            free(C_ref);
        }
    }

    free(rows_per_rank);
    free(row_offset);
    free(send_counts);
    free(send_displs);
    free(A_local);
    free(C_local);
    free(B);
    if (rank == 0) {
        free(A);
        free(C);
        if (verify) free(A_serial_copy);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
