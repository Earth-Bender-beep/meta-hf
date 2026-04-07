// Polynomial evaluation using Horner's method
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_DEGREE 100
#define NUM_EVALUATIONS 100000

double horner_method(double *coeffs, int degree, double x) {
    double result = coeffs[degree];
    for (int i = degree - 1; i >= 0; i--) {
        result = result * x + coeffs[i];
    }
    return result;
}

double naive_eval(double *coeffs, int degree, double x) {
    double result = 0.0;
    double x_power = 1.0;
    
    for (int i = 0; i <= degree; i++) {
        result += coeffs[i] * x_power;
        x_power *= x;
    }
    
    return result;
}

void polynomial_derivative(double *coeffs, int degree, double *deriv) {
    for (int i = 0; i < degree; i++) {
        deriv[i] = coeffs[i + 1] * (i + 1);
    }
}

double newton_raphson_poly(double *coeffs, int degree, double initial_guess, int max_iter) {
    double *deriv = (double*)malloc(degree * sizeof(double));
    polynomial_derivative(coeffs, degree, deriv);
    
    double x = initial_guess;
    
    for (int iter = 0; iter < max_iter; iter++) {
        double f = horner_method(coeffs, degree, x);
        double df = horner_method(deriv, degree - 1, x);
        
        if (fabs(df) < 1e-10) break;
        
        x = x - f / df;
    }
    
    free(deriv);
    return x;
}

int main() {
    int degree = 20;
    double *coeffs = (double*)malloc((degree + 1) * sizeof(double));
    
    unsigned int seed = 42;
    for (int i = 0; i <= degree; i++) {
        seed = seed * 1103515245 + 12345;
        coeffs[i] = ((seed & 0xFFFF) / (double)0xFFFF) * 2.0 - 1.0;
    }
    
    clock_t start = clock();
    
    double sum_horner = 0.0;
    for (int i = 0; i < NUM_EVALUATIONS; i++) {
        double x = (i / (double)NUM_EVALUATIONS) * 4.0 - 2.0;
        sum_horner += horner_method(coeffs, degree, x);
    }
    
    double sum_naive = 0.0;
    for (int i = 0; i < NUM_EVALUATIONS; i++) {
        double x = (i / (double)NUM_EVALUATIONS) * 4.0 - 2.0;
        sum_naive += naive_eval(coeffs, degree, x);
    }
    
    double root = newton_raphson_poly(coeffs, degree, 1.0, 50);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Polynomial evaluation: degree=%d, %d evaluations, %.6f seconds\n",
           degree, NUM_EVALUATIONS * 2, time_spent);
    printf("Horner sum: %.6f, Naive sum: %.6f\n", sum_horner, sum_naive);
    printf("Root found: %.6f\n", root);
    
    free(coeffs);
    return 0;
}
