// Logistic map - demonstrates chaotic behavior and bifurcation
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_R_VALUES 1000
#define NUM_ITERATIONS 500
#define TRANSIENT 400

void logistic_map(double r, double x0, double *results, int n) {
    double x = x0;
    
    for (int i = 0; i < TRANSIENT; i++) {
        x = r * x * (1.0 - x);
    }
    
    for (int i = 0; i < n; i++) {
        x = r * x * (1.0 - x);
        results[i] = x;
    }
}

double compute_lyapunov(double r, double x0, int n) {
    double x = x0;
    double sum = 0.0;
    
    for (int i = 0; i < n; i++) {
        x = r * x * (1.0 - x);
        sum += log(fabs(r * (1.0 - 2.0 * x)));
    }
    
    return sum / n;
}

int main() {
    double *results = (double*)malloc(NUM_ITERATIONS * sizeof(double));
    double *bifurcation_data = (double*)malloc(NUM_R_VALUES * 100 * sizeof(double));
    
    double x0 = 0.5;
    
    clock_t start = clock();
    
    int data_count = 0;
    
    for (int i = 0; i < NUM_R_VALUES; i++) {
        double r = 2.5 + (i / (double)NUM_R_VALUES) * 1.5;
        
        logistic_map(r, x0, results, NUM_ITERATIONS - TRANSIENT);
        
        for (int j = 0; j < 10 && data_count < NUM_R_VALUES * 100; j++) {
            bifurcation_data[data_count++] = results[j];
        }
    }
    
    double lyapunov_25 = compute_lyapunov(2.5, x0, 1000);
    double lyapunov_35 = compute_lyapunov(3.5, x0, 1000);
    double lyapunov_39 = compute_lyapunov(3.9, x0, 1000);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Logistic map: %d r-values, %d iterations each, %.6f seconds\n",
           NUM_R_VALUES, NUM_ITERATIONS, time_spent);
    printf("Lyapunov exponents: r=2.5: %.4f, r=3.5: %.4f, r=3.9: %.4f\n",
           lyapunov_25, lyapunov_35, lyapunov_39);
    
    free(results);
    free(bifurcation_data);
    
    return 0;
}
