#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef N
#define N 1024
#endif

#ifndef NSET
#define NSET 4
#endif

#ifndef NITER
#define NITER 10
#endif

/* row-major, set-major layout:
 *   M[s][i][j] == M[s*N*N + i*N + j]
 * Each set occupies a contiguous N*N block, so M + s*N*N points to set s.
 */
#define IDX(s, i, j) ((size_t)(s) * N * N + (size_t)(i) * N + (j))

static void init_matrices(double *A, double *B, double *C, double *D) {
    for (int s = 0; s < NSET; s++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[IDX(s, i, j)] = (double)(i + j + s) * 0.001;
                B[IDX(s, i, j)] = (double)(i - j + s) * 0.001;
                C[IDX(s, i, j)] = 0.0;
                D[IDX(s, i, j)] = 0.0;
            }
        }
    }
}

/* For each set s: C[s] = A[s] * B[s] + C[s] */
static void matmul(int s, const double *A, const double *B, double *C) {
        const double *As = A + (size_t)s * N * N;
        const double *Bs = B + (size_t)s * N * N;
        double       *Cs = C + (size_t)s * N * N;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double sum = Cs[i * N + j];
                for (int k = 0; k < N; k++) {
                    sum += As[i * N + k] * Bs[k * N + j];
                }
                Cs[i * N + j] = sum;
            }
        }
}
static void addmul(int s, double *dst, const double *src) {
        double *dsts = dst + (size_t)s * N * N;
        const double *srcs = src + (size_t)s * N * N;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                dsts[i * N + j] += srcs[i * N + j];
            }
        }
}

static double checksum(const double *C) {
    double s = 0.0;
    size_t total = (size_t)NSET * N * N;
    for (size_t i = 0; i < total; i++) {
        s += C[i];
    }
    return s;
}

int main(void) {
    int x, y;
    size_t bytes = (size_t)NSET * N * N * sizeof(double);
    double *A = (double *)malloc(bytes);
    double *B = (double *)malloc(bytes);
    double *C = (double *)malloc(bytes);
    double *D = (double *)malloc(bytes);
    if (!A || !B || !C || !D) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    init_matrices(A, B, C, D);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for(x=0; x<NITER; x++){
        for(y=0; y<NSET; y++){
            matmul(y, A, B, C);
            matmul(y, A, B, D);
            addmul(y, C, D);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double flops = 2.0 * 2.0 * (double)NITER * (double)NSET *
                   (double)N * (double)N * (double)N;

    printf("N        = %d\n", N);
    printf("NSET     = %d\n", NSET);
    printf("Elapsed  = %.6f sec\n", elapsed);
    printf("GFLOPS   = %.3f\n", flops / elapsed / 1e9);
    printf("Checksum = %.6e\n", checksum(C));

    free(A);
    free(B);
    free(C);
    free(D);
    return 0;
}
