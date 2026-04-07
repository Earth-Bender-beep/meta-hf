// Bisection method for root finding
// Bracketing method with guaranteed convergence
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TOL 1e-10
#define MAX_ITER 100
#define NUM_TRIALS 10000

// Test function: x^3 - 2*x - 5 (root near 2.09)
double f1(double x) {
    return x * x * x - 2.0 * x - 5.0;
}

// Test function: cos(x) - x (root near 0.739)
double f2(double x) {
    return cos(x) - x;
}

// Test function: e^x - 3*x^2 (multiple roots)
double f3(double x) {
    return exp(x) - 3.0 * x * x;
}

double bisection(double (*f)(double), double a, double b, double tol, int max_iter) {
    if (f(a) * f(b) >= 0) {
        return NAN;  // No root or multiple roots in interval
    }
    
    double c = a;
    for (int i = 0; i < max_iter; i++) {
        c = (a + b) / 2.0;
        double fc = f(c);
        
        if (fabs(fc) < tol || fabs(b - a) < tol) {
            return c;
        }
        
        if (f(a) * fc < 0) {
            b = c;
        } else {
            a = c;
        }
    }
    
    return c;
}

int main() {
    clock_t start = clock();
    
    double sum = 0.0;
    int found = 0;
    
    for (int iter = 0; iter < NUM_TRIALS; iter++) {
        double root;
        
        // Find different roots
        if (iter % 3 == 0) {
            root = bisection(f1, 2.0, 3.0, TOL, MAX_ITER);
        } else if (iter % 3 == 1) {
            root = bisection(f2, 0.0, 1.0, TOL, MAX_ITER);
        } else {
            root = bisection(f3, 0.0, 1.0, TOL, MAX_ITER);
        }
        
        if (!isnan(root)) {
            sum += root;
            found++;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Bisection method: %d trials, %.6f seconds\n",
           NUM_TRIALS, time_spent);
    printf("Found %d roots, average: %.10f\n", found, sum / found);
    
    return 0;
}
