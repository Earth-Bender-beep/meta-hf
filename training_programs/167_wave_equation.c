// 1D wave equation solver using finite difference method
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define GRID_SIZE 1000
#define NUM_STEPS 2000
#define DX 0.1
#define DT 0.01
#define C 1.0

void wave_step(double *u, double *u_prev, double *u_next, int n, double c, double dx, double dt) {
    double r = c * dt / dx;
    double r_sq = r * r;
    
    u_next[0] = 0.0;
    u_next[n - 1] = 0.0;
    
    for (int i = 1; i < n - 1; i++) {
        u_next[i] = 2.0 * u[i] - u_prev[i] + r_sq * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
    }
}

void initial_condition(double *u, int n, double dx) {
    for (int i = 0; i < n; i++) {
        double x = i * dx;
        u[i] = exp(-(x - 5.0) * (x - 5.0));
    }
}

double compute_energy(double *u, double *u_prev, int n, double dx, double dt) {
    double kinetic = 0.0;
    double potential = 0.0;
    
    for (int i = 1; i < n - 1; i++) {
        double velocity = (u[i] - u_prev[i]) / dt;
        kinetic += velocity * velocity;
        
        double du_dx = (u[i + 1] - u[i - 1]) / (2.0 * dx);
        potential += du_dx * du_dx;
    }
    
    return 0.5 * dx * (kinetic + potential);
}

int main() {
    int n = GRID_SIZE;
    
    double *u = (double*)calloc(n, sizeof(double));
    double *u_prev = (double*)calloc(n, sizeof(double));
    double *u_next = (double*)calloc(n, sizeof(double));
    
    initial_condition(u, n, DX);
    initial_condition(u_prev, n, DX);
    
    clock_t start = clock();
    
    double initial_energy = compute_energy(u, u_prev, n, DX, DT);
    
    for (int step = 0; step < NUM_STEPS; step++) {
        wave_step(u, u_prev, u_next, n, C, DX, DT);
        
        double *temp = u_prev;
        u_prev = u;
        u = u_next;
        u_next = temp;
    }
    
    double final_energy = compute_energy(u, u_prev, n, DX, DT);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Wave equation: grid=%d, steps=%d, %.6f seconds\n",
           n, NUM_STEPS, time_spent);
    printf("Energy: initial=%.6f, final=%.6f\n", initial_energy, final_energy);
    
    free(u);
    free(u_prev);
    free(u_next);
    
    return 0;
}
