#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef N
#define N 1024
#endif

#ifndef NSET
#define NSET 4
#endif

/* How many of the NSET sets go to the GPU. */
#ifndef NSET_GPU
#define NSET_GPU (NSET / 2)
#endif

#ifndef NITER
#define NITER 10
#endif

#define IDX(s, i, j) ((size_t)(s) * N * N + (size_t)(i) * N + (j))

#define CUDA_CHECK(cmd)                                                       \
    do {                                                                      \
        cudaError_t _e = (cmd);                                               \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_e));                                  \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(cmd)                                                     \
    do {                                                                      \
        cublasStatus_t _s = (cmd);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error %s:%d %d\n", __FILE__, __LINE__,    \
                    (int)_s);                                                 \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

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

    /* GPU buffers hold only the [0, NSET_GPU) prefix. */
    size_t g_elems = (size_t)NSET_GPU * N * N;
    size_t g_bytes = g_elems * sizeof(double);

    double *dA = NULL, *dB = NULL, *dC = NULL, *dD = NULL;
    cudaStream_t stream = 0;
    cublasHandle_t handle = NULL;

    if (NSET_GPU > 0) {
        CUDA_CHECK(cudaMalloc(&dA, g_bytes));
        CUDA_CHECK(cudaMalloc(&dB, g_bytes));
        CUDA_CHECK(cudaMalloc(&dC, g_bytes));
        CUDA_CHECK(cudaMalloc(&dD, g_bytes));
        CUDA_CHECK(cudaMemcpy(dA, A, g_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B, g_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dC, C, g_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dD, D, g_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetStream(handle, stream));
    }

    const double one = 1.0;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int x = 0; x < NITER; x++) {
        /* GPU side: cuBLAS is column-major. Our data is row-major, so by
         * swapping operands (B then A) the column-major result equals the
         * row-major C += A*B that we want.
         * Issued on `stream`, returns immediately on the host. */
        for (int y = 0; y < NSET_GPU; y++) {
            size_t off = (size_t)y * N * N;
            CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, N, N,
                                     &one, dB + off, N,
                                           dA + off, N,
                                     &one, dC + off, N));
            CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, N, N,
                                     &one, dB + off, N,
                                           dA + off, N,
                                     &one, dD + off, N));
            CUBLAS_CHECK(cublasDaxpy(handle, N * N, &one,
                                     dD + off, 1, dC + off, 1));
        }

        /* CPU side runs concurrently with the GPU stream. */
        for (int y = NSET_GPU; y < NSET; y++) {
            matmul_cpu(y, A, B, C);
            matmul_cpu(y, A, B, D);
            addmul_cpu(y, C, D);
        }

        if (NSET_GPU > 0) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    if (NSET_GPU > 0) {
        CUDA_CHECK(cudaMemcpy(C, dC, g_bytes, cudaMemcpyDeviceToHost));
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double flops = 2.0 * 2.0 * (double)NITER * (double)NSET *
                   (double)N * (double)N * (double)N;

    printf("[cuda+cublas+omp] N=%d NSET=%d (GPU=%d, CPU=%d) NITER=%d\n",
           N, NSET, NSET_GPU, NSET - NSET_GPU, NITER);
    printf("Elapsed  = %.6f sec\n", elapsed);
    printf("GFLOPS   = %.3f\n", flops / elapsed / 1e9);
    printf("Checksum = %.6e\n", checksum(C));

    if (NSET_GPU > 0) {
        cublasDestroy(handle);
        cudaStreamDestroy(stream);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        cudaFree(dD);
    }
    free(A);
    free(B);
    free(C);
    free(D);
    return 0;
}
