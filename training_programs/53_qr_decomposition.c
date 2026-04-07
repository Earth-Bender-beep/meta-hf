// QR decomposition using Gram-Schmidt orthogonalization
// Matrix factorization: A = Q*R where Q is orthogonal, R is upper triangular
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define M 128
#define N 128

void vector_copy(double *src, double *dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

double vector_dot(double *a, double *b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

double vector_norm(double *v, int n) {
    return sqrt(vector_dot(v, v, n));
}

void qr_decomposition(double *A, double *Q, double *R, int m, int n) {
    double *column = (double*)malloc(m * sizeof(double));
    double *temp = (double*)malloc(m * sizeof(double));
    
    // Initialize Q and R
    for (int i = 0; i < m * n; i++) Q[i] = 0.0;
    for (int i = 0; i < n * n; i++) R[i] = 0.0;
    
    for (int j = 0; j < n; j++) {
        // Extract column j of A
        for (int i = 0; i < m; i++) {
            column[i] = A[i * n + j];
        }
        
        // Subtract projections onto previous Q columns
        for (int k = 0; k < j; k++) {
            double proj = 0.0;
            for (int i = 0; i < m; i++) {
                proj += Q[i * n + k] * column[i];
            }
            R[k * n + j] = proj;
            
            for (int i = 0; i < m; i++) {
                column[i] -= proj * Q[i * n + k];
            }
        }
        
        // Normalize to get Q column
        double norm = vector_norm(column, m);
        R[j * n + j] = norm;
        
        if (norm > 1e-10) {
            for (int i = 0; i < m; i++) {
                Q[i * n + j] = column[i] / norm;
            }
        }
    }
    
    free(column);
    free(temp);
}

void matrix_multiply(double *A, double *B, double *C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

void init_matrix(double *A, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (double)((i + j * 3) % 20) / 10.0 - 1.0;
        }
    }
}

int main() {
    double *A = (double*)malloc(M * N * sizeof(double));
    double *Q = (double*)malloc(M * N * sizeof(double));
    double *R = (double*)malloc(N * N * sizeof(double));
    double *A_check = (double*)malloc(M * N * sizeof(double));
    
    init_matrix(A, M, N);
    
    clock_t start = clock();
    qr_decomposition(A, Q, R, M, N);
    clock_t end = clock();
    
    // Verify: Q*R should equal A
    matrix_multiply(Q, R, A_check, M, N, N);
    
    double error = 0.0;
    for (int i = 0; i < M * N; i++) {
        double diff = A[i] - A_check[i];
        error += diff * diff;
    }
    error = sqrt(error);
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("QR decomposition %dx%d: %.6f seconds, reconstruction_error=%.2e\n",
           M, N, time_spent, error);
    
    free(A); free(Q); free(R); free(A_check);
    return 0;
}
