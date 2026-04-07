// Cholesky decomposition for symmetric positive definite matrices
// Lower triangular factorization: A = L*L^T
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 256

int cholesky_decompose(double *A, double *L, int n) {
    for (int i = 0; i < n * n; i++) {
        L[i] = 0.0;
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            
            if (j == i) {
                for (int k = 0; k < j; k++) {
                    sum += L[j * n + k] * L[j * n + k];
                }
                double val = A[j * n + j] - sum;
                if (val <= 0.0) return 0; // Not positive definite
                L[j * n + j] = sqrt(val);
            } else {
                for (int k = 0; k < j; k++) {
                    sum += L[i * n + k] * L[j * n + k];
                }
                L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
            }
        }
    }
    return 1;
}

void forward_substitution(double *L, double *b, double *x, int n) {
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) {
            sum += L[i * n + j] * x[j];
        }
        x[i] = (b[i] - sum) / L[i * n + i];
    }
}

void backward_substitution(double *L, double *y, double *x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += L[j * n + i] * x[j];
        }
        x[i] = (y[i] - sum) / L[i * n + i];
    }
}

void make_spd_matrix(double *A, int n) {
    // Create symmetric positive definite matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double val = ((i + j) % 10) / 10.0;
            A[i * n + j] = val;
            A[j * n + i] = val;
        }
        A[i * n + i] += n; // Make diagonally dominant
    }
}

int main() {
    double *A = (double*)malloc(N * N * sizeof(double));
    double *L = (double*)malloc(N * N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *y = (double*)malloc(N * sizeof(double));
    double *x = (double*)malloc(N * sizeof(double));
    
    make_spd_matrix(A, N);
    for (int i = 0; i < N; i++) {
        b[i] = (double)(i % 10);
    }
    
    clock_t start = clock();
    
    if (cholesky_decompose(A, L, N)) {
        forward_substitution(L, b, y, N);
        backward_substitution(L, y, x, N);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Cholesky %dx%d: %.6f seconds, x[0]=%.6f\n", 
           N, N, time_spent, x[0]);
    
    free(A); free(L); free(b); free(y); free(x);
    return 0;
}
