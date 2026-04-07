// 2D Heat diffusion using finite differences
// Stencil computation, iterative solver, physics simulation
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 256
#define TIME_STEPS 500
#define ALPHA 0.1

void heat_diffusion_step(double *T, double *T_new, int n, double alpha) {
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            T_new[i * n + j] = T[i * n + j] + alpha * (
                T[(i-1) * n + j] + T[(i+1) * n + j] + 
                T[i * n + (j-1)] + T[i * n + (j+1)] - 
                4.0 * T[i * n + j]
            );
        }
    }
    
    // Copy boundaries
    for (int i = 0; i < n; i++) {
        T_new[i] = T[i];
        T_new[(n-1) * n + i] = T[(n-1) * n + i];
        T_new[i * n] = T[i * n];
        T_new[i * n + (n-1)] = T[i * n + (n-1)];
    }
}

void init_temperature(double *T, int n) {
    for (int i = 0; i < n * n; i++) {
        T[i] = 0.0;
    }
    
    // Hot spot in center
    int center = n / 2;
    int radius = n / 8;
    for (int i = center - radius; i < center + radius; i++) {
        for (int j = center - radius; j < center + radius; j++) {
            T[i * n + j] = 100.0;
        }
    }
    
    // Boundary conditions
    for (int i = 0; i < n; i++) {
        T[i] = 20.0;
        T[(n-1) * n + i] = 20.0;
        T[i * n] = 20.0;
        T[i * n + (n-1)] = 20.0;
    }
}

double compute_average_temp(double *T, int n) {
    double sum = 0.0;
    for (int i = 0; i < n * n; i++) {
        sum += T[i];
    }
    return sum / (n * n);
}

int main() {
    double *T = (double*)malloc(N * N * sizeof(double));
    double *T_new = (double*)malloc(N * N * sizeof(double));
    
    init_temperature(T, N);
    
    clock_t start = clock();
    
    for (int step = 0; step < TIME_STEPS; step++) {
        heat_diffusion_step(T, T_new, N, ALPHA);
        
        // Swap pointers
        double *temp = T;
        T = T_new;
        T_new = temp;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    double avg_temp = compute_average_temp(T, N);
    printf("Heat diffusion %dx%d (%d steps): %.6f seconds, avg_temp=%.2f\n",
           N, N, TIME_STEPS, time_spent, avg_temp);
    
    free(T);
    free(T_new);
    return 0;
}
