// Laplace equation solver using Gauss-Seidel iteration
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define GRID_SIZE 100
#define MAX_ITERATIONS 1000
#define TOLERANCE 1e-5

void initialize_grid(double **grid, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            grid[i][j] = 0.0;
        }
    }
    
    for (int i = 0; i < n; i++) {
        grid[0][i] = 100.0;
        grid[n - 1][i] = 0.0;
        grid[i][0] = 0.0;
        grid[i][n - 1] = 0.0;
    }
}

double gauss_seidel_iteration(double **grid, int n) {
    double max_change = 0.0;
    
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            double old_value = grid[i][j];
            
            grid[i][j] = 0.25 * (grid[i - 1][j] + grid[i + 1][j] +
                                 grid[i][j - 1] + grid[i][j + 1]);
            
            double change = fabs(grid[i][j] - old_value);
            if (change > max_change) {
                max_change = change;
            }
        }
    }
    
    return max_change;
}

double compute_average(double **grid, int n) {
    double sum = 0.0;
    int count = 0;
    
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            sum += grid[i][j];
            count++;
        }
    }
    
    return sum / count;
}

int main() {
    int n = GRID_SIZE;
    
    double **grid = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        grid[i] = (double*)malloc(n * sizeof(double));
    }
    
    initialize_grid(grid, n);
    
    clock_t start = clock();
    
    int iterations = 0;
    double change;
    
    do {
        change = gauss_seidel_iteration(grid, n);
        iterations++;
    } while (change > TOLERANCE && iterations < MAX_ITERATIONS);
    
    double average = compute_average(grid, n);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Laplace equation: %dx%d grid, %d iterations, %.6f seconds\n",
           n, n, iterations, time_spent);
    printf("Average potential: %.6f, Final change: %.10f\n", average, change);
    
    for (int i = 0; i < n; i++) {
        free(grid[i]);
    }
    free(grid);
    
    return 0;
}
