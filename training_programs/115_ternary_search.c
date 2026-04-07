// Ternary search for finding maximum/minimum of unimodal functions
// Divide and conquer optimization algorithm
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_TRIALS 50000
#define TOL 1e-9

// Unimodal test function 1: -x^2 + 4x - 1 (maximum at x=2)
double f1(double x) {
    return -x * x + 4.0 * x - 1.0;
}

// Unimodal test function 2: -(x-3)^4 + 10 (maximum at x=3)
double f2(double x) {
    double diff = x - 3.0;
    return -(diff * diff * diff * diff) + 10.0;
}

// Unimodal test function 3: -|x-5| + 8 (maximum at x=5)
double f3(double x) {
    return -fabs(x - 5.0) + 8.0;
}

double ternary_search_max(double (*f)(double), double left, double right, double tol) {
    while (right - left > tol) {
        double m1 = left + (right - left) / 3.0;
        double m2 = right - (right - left) / 3.0;
        
        if (f(m1) < f(m2)) {
            left = m1;
        } else {
            right = m2;
        }
    }
    
    return (left + right) / 2.0;
}

double ternary_search_min(double (*f)(double), double left, double right, double tol) {
    while (right - left > tol) {
        double m1 = left + (right - left) / 3.0;
        double m2 = right - (right - left) / 3.0;
        
        if (f(m1) > f(m2)) {
            left = m1;
        } else {
            right = m2;
        }
    }
    
    return (left + right) / 2.0;
}

int main() {
    clock_t start = clock();
    
    double sum = 0.0;
    
    for (int iter = 0; iter < NUM_TRIALS; iter++) {
        double result;
        
        if (iter % 3 == 0) {
            result = ternary_search_max(f1, 0.0, 4.0, TOL);
        } else if (iter % 3 == 1) {
            result = ternary_search_max(f2, 0.0, 6.0, TOL);
        } else {
            result = ternary_search_max(f3, 0.0, 10.0, TOL);
        }
        
        sum += result;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Ternary search: %d trials, %.6f seconds\n", NUM_TRIALS, time_spent);
    printf("Average result: %.10f\n", sum / NUM_TRIALS);
    
    return 0;
}
