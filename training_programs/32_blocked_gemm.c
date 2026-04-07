// Blocked GEMM for cache optimization
// Different loop structure than naive - tests loop tiling optimizations
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 128
#define BLOCK 16

void gemm_blocked(double *A, double *B, double *C, int n, int block_size) {
    for (int ii = 0; ii < n; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            for (int kk = 0; kk < n; kk += block_size) {
                // Inner blocked computation
                for (int i = ii; i < ii + block_size && i < n; i++) {
                    for (int j = jj; j < jj + block_size && j < n; j++) {
                        double sum = C[i * n + j];
                        for (int k = kk; k < kk + block_size && k < n; k++) {
                            sum += A[i * n + k] * B[k * n + j];
                        }
                        C[i * n + j] = sum;
                    }
                }
            }
        }
    }
}

void init_matrix(double *M, int n, int seed) {
    for (int i = 0; i < n * n; i++) {
        M[i] = ((i * seed) % 200) / 20.0;
    }
}

int main() {
    double *A = (double*)malloc(N * N * sizeof(double));
    double *B = (double*)malloc(N * N * sizeof(double));
    double *C = (double*)calloc(N * N, sizeof(double));
    
    init_matrix(A, N, 17);
    init_matrix(B, N, 23);
    
    clock_t start = clock();
    gemm_blocked(A, B, C, N, BLOCK);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Blocked GEMM %dx%d (block=%d): %.6f seconds, C[0][0]=%.2f\n", 
           N, N, BLOCK, time_spent, C[0]);
    
    free(A);
    free(B);
    free(C);
    return 0;
}
