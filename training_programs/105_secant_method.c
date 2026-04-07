// Secant method for root finding
// Iterative method using finite differences (no derivatives needed)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TOL 1e-10
#define MAX_ITER 50
#define NUM_TRIALS 15000

// Test functions
double f1(double x) {
    return x * x * x - x - 2.0;  // Root near 1.52
}

double f2(double x) {
    return exp(x) - 3.0 * x;  // Root near 0.619
}

double f3(double x) {
    return sin(x) - 0.5 * x;  // Root near 1.896
}

double secant(double (*f)(double), double x0, double x1, double tol, int max_iter) {
    for (int i = 0; i < max_iter; i++) {
        double f0 = f(x0);
        double f1_val = f(x1);
        
        if (fabs(f1_val) < tol) {
            return x1;
        }
        
        if (fabs(f1_val - f0) < 1e-15) {
            return NAN;  // Division by zero
        }
        
        double x2 = x1 - f1_val * (x1 - x0) / (f1_val - f0);
        
        if (fabs(x2 - x1) < tol) {
            return x2;
        }
        
        x0 = x1;
        x1 = x2;
    }
    
    return x1;
}

int main() {
    clock_t start = clock();
    
    double sum = 0.0;
    int found = 0;
    
    for (int iter = 0; iter < NUM_TRIALS; iter++) {
        double root;
        
        if (iter % 3 == 0) {
            root = secant(f1, 1.0, 2.0, TOL, MAX_ITER);
        } else if (iter % 3 == 1) {
            root = secant(f2, 0.0, 1.0, TOL, MAX_ITER);
        } else {
            root = secant(f3, 1.0, 2.0, TOL, MAX_ITER);
        }
        
        if (!isnan(root)) {
            sum += root;
            found++;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Secant method: %d trials, %.6f seconds\n",
           NUM_TRIALS, time_spent);
    printf("Found %d roots, average: %.10f\n", found, sum / found);
    
    return 0;
}
