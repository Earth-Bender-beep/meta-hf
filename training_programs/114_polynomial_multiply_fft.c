// Polynomial multiplication using FFT
// Fast O(n log n) multiplication via Fast Fourier Transform
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>

#define MAX_DEG 4096
#define PI 3.14159265358979323846

void fft(double complex *a, int n, int inv) {
    if (n == 1) return;
    
    double complex *even = (double complex*)malloc(n/2 * sizeof(double complex));
    double complex *odd = (double complex*)malloc(n/2 * sizeof(double complex));
    
    for (int i = 0; i < n/2; i++) {
        even[i] = a[2*i];
        odd[i] = a[2*i + 1];
    }
    
    fft(even, n/2, inv);
    fft(odd, n/2, inv);
    
    double angle = 2 * PI / n * (inv ? -1 : 1);
    double complex w = 1;
    double complex wn = cexp(I * angle);
    
    for (int i = 0; i < n/2; i++) {
        a[i] = even[i] + w * odd[i];
        a[i + n/2] = even[i] - w * odd[i];
        if (inv) {
            a[i] /= 2;
            a[i + n/2] /= 2;
        }
        w *= wn;
    }
    
    free(even);
    free(odd);
}

void polynomial_multiply(double *a, int deg_a, double *b, int deg_b, double *result) {
    int result_deg = deg_a + deg_b;
    
    // Find next power of 2
    int n = 1;
    while (n <= result_deg) n <<= 1;
    
    double complex *fa = (double complex*)calloc(n, sizeof(double complex));
    double complex *fb = (double complex*)calloc(n, sizeof(double complex));
    
    for (int i = 0; i <= deg_a; i++) fa[i] = a[i];
    for (int i = 0; i <= deg_b; i++) fb[i] = b[i];
    
    fft(fa, n, 0);
    fft(fb, n, 0);
    
    for (int i = 0; i < n; i++) {
        fa[i] *= fb[i];
    }
    
    fft(fa, n, 1);
    
    for (int i = 0; i <= result_deg; i++) {
        result[i] = creal(fa[i]);
    }
    
    free(fa);
    free(fb);
}

void generate_polynomial(double *poly, int degree) {
    unsigned int seed = 42;
    for (int i = 0; i <= degree; i++) {
        seed = seed * 1103515245 + 12345;
        poly[i] = (double)((seed % 100) - 50);
    }
}

int main() {
    int deg = 2000;
    
    double *poly_a = (double*)malloc((deg + 1) * sizeof(double));
    double *poly_b = (double*)malloc((deg + 1) * sizeof(double));
    double *result = (double*)calloc(2 * deg + 1, sizeof(double));
    
    generate_polynomial(poly_a, deg);
    generate_polynomial(poly_b, deg);
    
    clock_t start = clock();
    polynomial_multiply(poly_a, deg, poly_b, deg, result);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Checksum
    double sum = 0.0;
    for (int i = 0; i <= 2 * deg; i++) {
        sum += result[i];
    }
    
    printf("Polynomial multiply (FFT): degree=%d, %.6f seconds\n", deg, time_spent);
    printf("Result sum: %.2f\n", sum);
    
    free(poly_a);
    free(poly_b);
    free(result);
    
    return 0;
}
