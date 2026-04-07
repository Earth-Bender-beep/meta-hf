// Numerical integration using Simpson's rule
// Function evaluation heavy, low memory, moderate arithmetic
#include <stdio.h>
#include <math.h>
#include <time.h>

#define N_INTERVALS 100000

// Test functions
double f1(double x) {
    return sin(x) * sin(x) + cos(x * 2.0);
}

double f2(double x) {
    return exp(-x * x / 2.0) / sqrt(2.0 * M_PI);
}

double f3(double x) {
    return 1.0 / (1.0 + x * x);
}

double f4(double x) {
    return x * x * x - 2.0 * x * x + 3.0 * x - 1.0;
}

double f5(double x) {
    return log(1.0 + x) / (1.0 + x * x);
}

// Simpson's 1/3 rule
double simpson_integrate(double (*f)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = f(a) + f(b);
    
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        if (i % 2 == 0) {
            sum += 2.0 * f(x);
        } else {
            sum += 4.0 * f(x);
        }
    }
    
    return (h / 3.0) * sum;
}

// Adaptive Simpson's rule for variable accuracy
double adaptive_simpson(double (*f)(double), double a, double b, double eps) {
    double c = (a + b) / 2.0;
    double h = b - a;
    double fa = f(a), fb = f(b), fc = f(c);
    
    double s1 = (h / 6.0) * (fa + 4.0 * fc + fb);
    double s2 = (h / 12.0) * (fa + 4.0 * f((a + c) / 2.0) + 2.0 * fc + 
                               4.0 * f((c + b) / 2.0) + fb);
    
    if (fabs(s2 - s1) < 15.0 * eps) {
        return s2 + (s2 - s1) / 15.0;
    }
    
    return adaptive_simpson(f, a, c, eps / 2.0) + 
           adaptive_simpson(f, c, b, eps / 2.0);
}

// Composite trapezoidal rule
double trapezoidal_integrate(double (*f)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.5 * (f(a) + f(b));
    
    for (int i = 1; i < n; i++) {
        sum += f(a + i * h);
    }
    
    return h * sum;
}

int main() {
    clock_t start = clock();
    
    // Integrate multiple functions
    double result1 = simpson_integrate(f1, 0.0, M_PI, N_INTERVALS);
    double result2 = simpson_integrate(f2, -3.0, 3.0, N_INTERVALS);
    double result3 = simpson_integrate(f3, 0.0, 1.0, N_INTERVALS);
    double result4 = trapezoidal_integrate(f4, -1.0, 2.0, N_INTERVALS);
    double result5 = trapezoidal_integrate(f5, 0.0, 1.0, N_INTERVALS);
    
    // Some adaptive integration
    double result6 = adaptive_simpson(f1, 0.0, M_PI / 2.0, 1e-6);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Numerical integration (%d intervals): %.6f seconds\n", 
           N_INTERVALS, time_spent);
    printf("Results: %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n",
           result1, result2, result3, result4, result5, result6);
    
    return 0;
}
