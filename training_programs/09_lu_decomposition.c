#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void lu_decomposition(double** A, double** L, double** U, int n) {
    for (int i = 0; i < n; i++) {
        for (int k = i; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += (L[i][j] * U[j][k]);
            U[i][k] = A[i][k] - sum;
        }
        
        for (int k = i; k < n; k++) {
            if (i == k)
                L[i][i] = 1;
            else {
                double sum = 0;
                for (int j = 0; j < i; j++)
                    sum += (L[k][j] * U[j][i]);
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }
}

int main() {
    int n = 400;
    
    double** A = (double**)malloc(n * sizeof(double*));
    double** L = (double**)malloc(n * sizeof(double*));
    double** U = (double**)malloc(n * sizeof(double*));
    
    for (int i = 0; i < n; i++) {
        A[i] = (double*)malloc(n * sizeof(double));
        L[i] = (double*)calloc(n, sizeof(double));
        U[i] = (double*)calloc(n, sizeof(double));
    }
    
    srand(42);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (double)(rand() % 100) / 10.0;
        }
        A[i][i] += n;
    }
    
    clock_t start = clock();
    lu_decomposition(A, L, U, n);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("LU decomposition: %dx%d in %.6f seconds\n", n, n, time_spent);
    
    for (int i = 0; i < n; i++) {
        free(A[i]);
        free(L[i]);
        free(U[i]);
    }
    free(A);
    free(L);
    free(U);
    
    return 0;
}
