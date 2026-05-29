#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <openacc.h>

#ifndef N
#define N 1024
#endif

#ifndef NSET
#define NSET 4
#endif

#ifndef NSET_GPU
#define NSET_GPU (NSET / 2)
#endif

#ifndef NITER
#define NITER 10
#endif

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

static void matmul_gpu(int s, const double *A, const double *B, double *C) {
    const double *As = A + (size_t)s * N * N;
    const double *Bs = B + (size_t)s * N * N;
    double       *Cs = C + (size_t)s * N * N;
    #pragma acc parallel loop collapse(2) async(1) \
        present(As[0:N*N], Bs[0:N*N], Cs[0:N*N])
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = Cs[i * N + j];
            #pragma acc loop seq
            for (int k = 0; k < N; k++) {
                sum += As[i * N + k] * Bs[k * N + j];
            }
            Cs[i * N + j] = sum;
        }
    }
}

static void addmul_gpu(int s, double *dst, const double *src) {
    double *dsts = dst + (size_t)s * N * N;
    const double *srcs = src + (size_t)s * N * N;
    #pragma acc parallel loop collapse(2) async(1) \
        present(dsts[0:N*N], srcs[0:N*N])
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            dsts[i * N + j] += srcs[i * N + j];
        }
    }
}

static void matmul_cpu(int s, const double *A, const double *B, double *C) {
    const double *As = A + (size_t)s * N * N;
    const double *Bs = B + (size_t)s * N * N;
    double       *Cs = C + (size_t)s * N * N;
    #pragma omp parallel for collapse(2) schedule(static)
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

static void addmul_cpu(int s, double *dst, const double *src) {
    double *dsts = dst + (size_t)s * N * N;
    const double *srcs = src + (size_t)s * N * N;
    #pragma omp parallel for collapse(2) schedule(static)
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

    size_t g_elems = (size_t)NSET_GPU * N * N;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if (NSET_GPU > 0) {
        #pragma acc enter data copyin(A[0:g_elems], B[0:g_elems], \
                                      C[0:g_elems], D[0:g_elems])
    }

    /* GPU制御スレッドとCPU計算スレッドを分離する。
     * - GPU側は全反復・全GPUセットを async(1) で投入し、最後に1回だけ wait。
     *   wait でブロックされるのはこのスレッドだけなので、複雑な計算で
     *   途中の同期が必要になっても CPU 計算とは真に並列に進む。
     * - GPU/CPU のセットはメモリが独立なので、毎反復のバリアは不要。
     * 内側の matmul_cpu は omp parallel for を持つため入れ子並列を有効化する。 */
    omp_set_max_active_levels(2);

    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            for (int x = 0; x < NITER; x++) {
                for (int y = 0; y < NSET_GPU; y++) {
                    matmul_gpu(y, A, B, C);
                    matmul_gpu(y, A, B, D);
                    addmul_gpu(y, C, D);
                }
            }
            if (NSET_GPU > 0) {
                #pragma acc wait(1)
            }
        }

        #pragma omp section
        {
            for (int x = 0; x < NITER; x++) {
                for (int y = NSET_GPU; y < NSET; y++) {
                    matmul_cpu(y, A, B, C);
                    matmul_cpu(y, A, B, D);
                    addmul_cpu(y, C, D);
                }
            }
        }
    }

    if (NSET_GPU > 0) {
        #pragma acc exit data copyout(C[0:g_elems]) \
                              delete(A[0:g_elems], B[0:g_elems], D[0:g_elems])
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double flops = 2.0 * 2.0 * (double)NITER * (double)NSET *
                   (double)N * (double)N * (double)N;

    printf("[acc+omp v2:decoupled] N=%d NSET=%d (GPU=%d, CPU=%d) NITER=%d\n",
           N, NSET, NSET_GPU, NSET - NSET_GPU, NITER);
    printf("Elapsed  = %.6f sec\n", elapsed);
    printf("GFLOPS   = %.3f\n", flops / elapsed / 1e9);
    printf("Checksum = %.6e\n", checksum(C));

    free(A);
    free(B);
    free(C);
    free(D);
    return 0;
}
