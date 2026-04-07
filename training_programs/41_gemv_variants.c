// Matrix-vector multiply with different access patterns
// Tests memory access optimization (row vs column major)
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 512
#define N 512

// Row-major GEMV: y = A*x
void gemv_row_major(double *A, double *x, double *y, int m, int n) {
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

// Column-major GEMV: y = A*x
void gemv_col_major(double *A, double *x, double *y, int m, int n) {
    for (int i = 0; i < m; i++) {
        y[i] = 0.0;
    }
    
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

// Blocked GEMV for cache optimization
void gemv_blocked(double *A, double *x, double *y, int m, int n, int block) {
    for (int i = 0; i < m; i++) {
        y[i] = 0.0;
    }
    
    for (int jj = 0; jj < n; jj += block) {
        int j_end = (jj + block < n) ? jj + block : n;
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            for (int j = jj; j < j_end; j++) {
                sum += A[i * n + j] * x[j];
            }
            y[i] += sum;
        }
    }
}

void init_data(double *A, double *x, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        A[i] = (i % 100) / 50.0 - 1.0;
    }
    for (int i = 0; i < n; i++) {
        x[i] = (i % 50) / 25.0;
    }
}

int main() {
    double *A = (double*)malloc(M * N * sizeof(double));
    double *x = (double*)malloc(N * sizeof(double));
    double *y1 = (double*)malloc(M * sizeof(double));
    double *y2 = (double*)malloc(M * sizeof(double));
    double *y3 = (double*)malloc(M * sizeof(double));
    
    init_data(A, x, M, N);
    
    clock_t start = clock();
    gemv_row_major(A, x, y1, M, N);
    gemv_col_major(A, x, y2, M, N);
    gemv_blocked(A, x, y3, M, N, 32);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("GEMV %dx%d variants: %.6f seconds, y1[0]=%.4f, y2[0]=%.4f, y3[0]=%.4f\n",
           M, N, time_spent, y1[0], y2[0], y3[0]);
    
    free(A); free(x); free(y1); free(y2); free(y3);
    return 0;
}
