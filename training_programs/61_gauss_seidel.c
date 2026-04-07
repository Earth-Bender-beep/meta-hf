// Gauss-Seidel iterative method for solving linear systems
// Iterative solver for Ax = b, uses updated values immediately
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 256
#define MAX_ITER 500
#define TOLERANCE 1e-6

double gauss_seidel(double *A, double *b, double *x, int n, int max_iter, double tol) {
    double *x_old = (double*)malloc(n * sizeof(double));
    int iter;
    
    // Initialize x
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
    }
    
    for (iter = 0; iter < max_iter; iter++) {
        // Save old values
        for (int i = 0; i < n; i++) {
            x_old[i] = x[i];
        }
        
        // Gauss-Seidel iteration
        for (int i = 0; i < n; i++) {
            double sum = b[i];
            
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    sum -= A[i * n + j] * x[j];
                }
            }
            
            x[i] = sum / A[i * n + i];
        }
        
        // Check convergence
        double max_diff = 0.0;
        for (int i = 0; i < n; i++) {
            double diff = fabs(x[i] - x_old[i]);
            if (diff > max_diff) max_diff = diff;
        }
        
        if (max_diff < tol) {
            break;
        }
    }
    
    free(x_old);
    return (double)iter;
}

void create_diagonally_dominant_system(double *A, double *b, int n) {
    // Create a diagonally dominant system (guarantees convergence)
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                A[i * n + j] = ((i + j) % 10) / 20.0;
                row_sum += fabs(A[i * n + j]);
            }
        }
        A[i * n + i] = row_sum + 5.0;  // Make diagonally dominant
        b[i] = (double)(i % 10);
    }
}

int main() {
    double *A = (double*)malloc(N * N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *x = (double*)malloc(N * sizeof(double));
    
    create_diagonally_dominant_system(A, b, N);
    
    clock_t start = clock();
    double iterations = gauss_seidel(A, b, x, N, MAX_ITER, TOLERANCE);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Gauss-Seidel %dx%d: %.6f seconds, %d iterations, x[0]=%.6f\n",
           N, N, time_spent, (int)iterations, x[0]);
    
    free(A);
    free(b);
    free(x);
    return 0;
}
