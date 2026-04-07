// Cooley-Tukey FFT algorithm (radix-2)
// Divide-and-conquer, butterfly operations, bit-reversal
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 4096

typedef struct {
    double real;
    double imag;
} Complex;

void fft_swap(Complex *a, Complex *b) {
    Complex temp = *a;
    *a = *b;
    *b = temp;
}

unsigned int reverse_bits(unsigned int x, int log_n) {
    unsigned int result = 0;
    for (int i = 0; i < log_n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

void fft_bit_reverse(Complex *data, int n) {
    int log_n = 0;
    int temp = n;
    while (temp > 1) {
        log_n++;
        temp >>= 1;
    }
    
    for (unsigned int i = 0; i < n; i++) {
        unsigned int j = reverse_bits(i, log_n);
        if (i < j) {
            fft_swap(&data[i], &data[j]);
        }
    }
}

void fft_radix2(Complex *data, int n) {
    fft_bit_reverse(data, n);
    
    for (int stage = 1; stage <= log2(n); stage++) {
        int m = 1 << stage;
        int m2 = m >> 1;
        
        double theta = -2.0 * M_PI / m;
        Complex wm = {cos(theta), sin(theta)};
        
        for (int k = 0; k < n; k += m) {
            Complex w = {1.0, 0.0};
            
            for (int j = 0; j < m2; j++) {
                Complex t;
                t.real = w.real * data[k + j + m2].real - w.imag * data[k + j + m2].imag;
                t.imag = w.real * data[k + j + m2].imag + w.imag * data[k + j + m2].real;
                
                Complex u = data[k + j];
                
                data[k + j].real = u.real + t.real;
                data[k + j].imag = u.imag + t.imag;
                
                data[k + j + m2].real = u.real - t.real;
                data[k + j + m2].imag = u.imag - t.imag;
                
                double w_real = w.real * wm.real - w.imag * wm.imag;
                double w_imag = w.real * wm.imag + w.imag * wm.real;
                w.real = w_real;
                w.imag = w_imag;
            }
        }
    }
}

void init_signal(Complex *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i].real = cos(2.0 * M_PI * 5.0 * i / n) + 
                       0.5 * cos(2.0 * M_PI * 13.0 * i / n);
        data[i].imag = 0.0;
    }
}

int main() {
    Complex *data = (Complex*)malloc(N * sizeof(Complex));
    
    init_signal(data, N);
    
    clock_t start = clock();
    fft_radix2(data, N);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    double max_mag = 0.0;
    int max_idx = 0;
    for (int i = 0; i < N / 2; i++) {
        double mag = sqrt(data[i].real * data[i].real + data[i].imag * data[i].imag);
        if (mag > max_mag) {
            max_mag = mag;
            max_idx = i;
        }
    }
    
    printf("FFT radix-2 (N=%d): %.6f seconds, peak at bin %d (mag=%.2f)\n",
           N, time_spent, max_idx, max_mag);
    
    free(data);
    return 0;
}
