#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846

typedef struct Complex {
    double real;
    double imag;
} Complex;

Complex add(Complex a, Complex b) {
    Complex result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

Complex sub(Complex a, Complex b) {
    Complex result;
    result.real = a.real - b.real;
    result.imag = a.imag - b.imag;
    return result;
}

Complex mul(Complex a, Complex b) {
    Complex result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

void fft(Complex* x, int n) {
    if (n <= 1) return;
    
    Complex* even = (Complex*)malloc((n / 2) * sizeof(Complex));
    Complex* odd = (Complex*)malloc((n / 2) * sizeof(Complex));
    
    for (int i = 0; i < n / 2; i++) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }
    
    fft(even, n / 2);
    fft(odd, n / 2);
    
    for (int k = 0; k < n / 2; k++) {
        Complex t;
        t.real = cos(-2 * PI * k / n);
        t.imag = sin(-2 * PI * k / n);
        t = mul(t, odd[k]);
        
        x[k] = add(even[k], t);
        x[k + n / 2] = sub(even[k], t);
    }
    
    free(even);
    free(odd);
}

int main() {
    int n = 8192;
    Complex* data = (Complex*)malloc(n * sizeof(Complex));
    
    srand(42);
    for (int i = 0; i < n; i++) {
        data[i].real = (double)(rand() % 100) / 10.0;
        data[i].imag = 0.0;
    }
    
    clock_t start = clock();
    fft(data, n);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("FFT: n=%d in %.6f seconds\n", n, time_spent);
    
    free(data);
    return 0;
}
