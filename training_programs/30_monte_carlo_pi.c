#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double estimate_pi(long long iterations) {
    long long inside_circle = 0;
    
    for (long long i = 0; i < iterations; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        
        if (x * x + y * y <= 1.0)
            inside_circle++;
    }
    
    return 4.0 * inside_circle / iterations;
}

int main() {
    long long iterations = 50000000;
    
    srand(42);
    clock_t start = clock();
    double pi = estimate_pi(iterations);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Monte Carlo Pi: %lld iterations, pi≈%.6f in %.6f seconds\n", 
           iterations, pi, time_spent);
    
    return 0;
}
