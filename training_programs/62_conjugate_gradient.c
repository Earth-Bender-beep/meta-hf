// Conjugate Gradient method for symmetric positive-definite systems
// Iterative Krylov subspace method, very efficient for sparse SPD matrices
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 300
#define MAX_ITER 500
#define TOLERANCE 1e-8

void matvec(double *A, double *x, double *y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;
        for (int j = 0; j < n; j++) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

double dot_product(double *a, double *b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

int conjugate_gradient(double *A, double *b, double *x, int n, int max_iter, double tol) {
    double *r = (double*)malloc(n * sizeof(double));
    double *p = (double*)malloc(n * sizeof(double));
    double *Ap = (double*)malloc(n * sizeof(double));
    
    // r = b - Ax
    matvec(A, x, r, n);
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - r[i];
        p[i] = r[i];
    }
    
    double rsold = dot_product(r, r, n);
    
    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        matvec(A, p, Ap, n);
        
        double pAp = dot_product(p, Ap, n);
        double alpha = rsold / pAp;
        
        // Update x and r
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        
        double rsnew = dot_product(r, r, n);
        
        if (sqrt(rsnew) < tol) {
            iter++;
            break;
        }
        
        double beta = rsnew / rsold;
        
        // Update p
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }
        
        rsold = rsnew;
    }
    
    free(r);
    free(p);
    free(Ap);
    
    return iter;
}

void create_spd_system(double *A, double *b, int n) {
    // Create symmetric positive-definite matrix
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double val = ((i + j) % 10) / 20.0;
            A[i * n + j] = val;
            A[j * n + i] = val;
        }
        A[i * n + i] += 10.0;  // Ensure positive definite
        b[i] = (double)(i % 10 + 1);
    }
}

int main() {
    double *A = (double*)malloc(N * N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *x = (double*)malloc(N * sizeof(double));
    
    // Initialize x to zero
    for (int i = 0; i < N; i++) {
        x[i] = 0.0;
    }
    
    create_spd_system(A, b, N);
    
    clock_t start = clock();
    int iterations = conjugate_gradient(A, b, x, N, MAX_ITER, TOLERANCE);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Conjugate Gradient %dx%d: %.6f seconds, %d iterations\n",
           N, N, time_spent, iterations);
    printf("Solution: x[0]=%.6f, x[%d]=%.6f\n", x[0], N-1, x[N-1]);
    
    free(A);
    free(b);
    free(x);
    return 0;
}
