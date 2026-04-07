// Newton-Raphson method for root finding
// Iterative numerical method, derivative evaluation
#include <stdio.h>
#include <math.h>
#include <time.h>

#define MAX_ITER 1000
#define TOLERANCE 1e-10

// f(x) = x^3 - 2x - 5
double f1(double x) {
    return x*x*x - 2.0*x - 5.0;
}

double f1_derivative(double x) {
    return 3.0*x*x - 2.0;
}

// f(x) = e^x - 3x
double f2(double x) {
    return exp(x) - 3.0*x;
}

double f2_derivative(double x) {
    return exp(x) - 3.0;
}

// f(x) = sin(x) - x/2
double f3(double x) {
    return sin(x) - x/2.0;
}

double f3_derivative(double x) {
    return cos(x) - 0.5;
}

// f(x) = x^4 - x - 10
double f4(double x) {
    return x*x*x*x - x - 10.0;
}

double f4_derivative(double x) {
    return 4.0*x*x*x - 1.0;
}

double newton_raphson(double (*f)(double), double (*df)(double), double x0, double tol, int max_iter, int *iterations) {
    double x = x0;
    *iterations = 0;
    
    for (int i = 0; i < max_iter; i++) {
        double fx = f(x);
        double dfx = df(x);
        
        if (fabs(dfx) < 1e-15) {
            break; // Derivative too small
        }
        
        double x_new = x - fx / dfx;
        (*iterations)++;
        
        if (fabs(x_new - x) < tol) {
            return x_new;
        }
        
        x = x_new;
    }
    
    return x;
}

// Multi-dimensional Newton (2D system)
void newton_2d(double x0, double y0, double *x_out, double *y_out, int max_iter) {
    double x = x0, y = y0;
    
    for (int i = 0; i < max_iter; i++) {
        // System: x^2 + y^2 - 4 = 0, x*y - 1 = 0
        double f1 = x*x + y*y - 4.0;
        double f2 = x*y - 1.0;
        
        // Jacobian
        double j11 = 2.0*x;
        double j12 = 2.0*y;
        double j21 = y;
        double j22 = x;
        
        double det = j11*j22 - j12*j21;
        if (fabs(det) < 1e-15) break;
        
        double dx = (j22*f1 - j12*f2) / det;
        double dy = (-j21*f1 + j11*f2) / det;
        
        x -= dx;
        y -= dy;
        
        if (fabs(dx) < TOLERANCE && fabs(dy) < TOLERANCE) {
            break;
        }
    }
    
    *x_out = x;
    *y_out = y;
}

int main() {
    clock_t start = clock();
    
    int iter1, iter2, iter3, iter4;
    double root1 = newton_raphson(f1, f1_derivative, 2.0, TOLERANCE, MAX_ITER, &iter1);
    double root2 = newton_raphson(f2, f2_derivative, 1.0, TOLERANCE, MAX_ITER, &iter2);
    double root3 = newton_raphson(f3, f3_derivative, 2.0, TOLERANCE, MAX_ITER, &iter3);
    double root4 = newton_raphson(f4, f4_derivative, 2.0, TOLERANCE, MAX_ITER, &iter4);
    
    double x2d, y2d;
    newton_2d(1.5, 1.5, &x2d, &y2d, MAX_ITER);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Newton-Raphson method: %.6f seconds\n", time_spent);
    printf("Root 1: %.10f (%d iter)\n", root1, iter1);
    printf("Root 2: %.10f (%d iter)\n", root2, iter2);
    printf("Root 3: %.10f (%d iter)\n", root3, iter3);
    printf("Root 4: %.10f (%d iter)\n", root4, iter4);
    printf("2D system: (%.6f, %.6f)\n", x2d, y2d);
    
    return 0;
}
