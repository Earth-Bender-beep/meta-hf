// Romberg integration using Richardson extrapolation
// Numerical integration with adaptive refinement
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_K 12

// Test function: sin(x) * exp(-x/10)
double test_function(double x) {
    return sin(x) * exp(-x / 10.0);
}

double trapezoidal(double (*f)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = (f(a) + f(b)) / 2.0;
    
    for (int i = 1; i < n; i++) {
        sum += f(a + i * h);
    }
    
    return h * sum;
}

double romberg_integrate(double (*f)(double), double a, double b, int max_k) {
    double R[MAX_K][MAX_K];
    
    // First column: trapezoidal rule with increasing subdivisions
    for (int i = 0; i < max_k; i++) {
        int n = 1 << i;  // 2^i
        R[i][0] = trapezoidal(f, a, b, n);
    }
    
    // Richardson extrapolation
    for (int j = 1; j < max_k; j++) {
        for (int i = j; i < max_k; i++) {
            double factor = (1 << (2 * j)) - 1;  // 4^j - 1
            R[i][j] = (factor * R[i][j-1] - R[i-1][j-1]) / factor;
        }
    }
    
    return R[max_k-1][max_k-1];
}

int main() {
    double a = 0.0;
    double b = 10.0;
    
    clock_t start = clock();
    
    double result = 0.0;
    for (int iter = 0; iter < 5000; iter++) {
        result += romberg_integrate(test_function, a, b, MAX_K);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Romberg integration: [%.1f, %.1f], max_k=%d, 5000 iterations, %.6f seconds\n",
           a, b, MAX_K, time_spent);
    printf("Average result: %.10f\n", result / 5000.0);
    
    return 0;
}
