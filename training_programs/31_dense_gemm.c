// Dense General Matrix Multiply (GEMM): C = alpha*A*B + beta*C
// High arithmetic intensity, low branching
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 128
#define ALPHA 1.5
#define BETA 0.5

void gemm_naive(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = ALPHA * sum + BETA * C[i * n + j];
        }
    }
}

void init_matrix(double *M, int n) {
    for (int i = 0; i < n * n; i++) {
        M[i] = (double)(i % 100) / 10.0;
    }
}

int main() {
    double *A = (double*)malloc(N * N * sizeof(double));
    double *B = (double*)malloc(N * N * sizeof(double));
    double *C = (double*)malloc(N * N * sizeof(double));
    
    init_matrix(A, N);
    init_matrix(B, N);
    init_matrix(C, N);
    
    clock_t start = clock();
    gemm_naive(A, B, C, N);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("GEMM %dx%d: %.6f seconds, result[0][0] = %.2f\n", 
           N, N, time_spent, C[0]);
    
    free(A);
    free(B);
    free(C);
    return 0;
}
