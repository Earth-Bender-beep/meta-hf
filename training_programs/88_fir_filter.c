// FIR (Finite Impulse Response) filter implementation
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SIGNAL_LENGTH 20000
#define FILTER_ORDER 64

void design_lowpass_fir(double *coeffs, int order, double cutoff) {
    int M = order;
    for (int n = 0; n <= M; n++) {
        int m = n - M / 2;
        if (m == 0) {
            coeffs[n] = 2.0 * cutoff;
        } else {
            coeffs[n] = sin(2.0 * M_PI * cutoff * m) / (M_PI * m);
        }
        
        // Apply Hamming window
        coeffs[n] *= 0.54 - 0.46 * cos(2.0 * M_PI * n / M);
    }
}

void fir_filter(double *input, double *output, double *coeffs, int signal_len, int filter_len) {
    for (int n = 0; n < signal_len; n++) {
        output[n] = 0.0;
        for (int k = 0; k < filter_len; k++) {
            if (n - k >= 0) {
                output[n] += coeffs[k] * input[n - k];
            }
        }
    }
}

int main() {
    double *signal = (double*)malloc(SIGNAL_LENGTH * sizeof(double));
    double *filtered = (double*)malloc(SIGNAL_LENGTH * sizeof(double));
    double *coeffs = (double*)malloc((FILTER_ORDER + 1) * sizeof(double));
    
    unsigned int seed = 42;
    for (int i = 0; i < SIGNAL_LENGTH; i++) {
        seed = seed * 1103515245 + 12345;
        signal[i] = sin(2.0 * M_PI * 0.01 * i) + 
                    0.5 * sin(2.0 * M_PI * 0.25 * i) +
                    0.2 * ((seed & 0xFFFF) / (double)0xFFFF - 0.5);
    }
    
    design_lowpass_fir(coeffs, FILTER_ORDER, 0.1);
    
    clock_t start = clock();
    fir_filter(signal, filtered, coeffs, SIGNAL_LENGTH, FILTER_ORDER + 1);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("FIR filter: signal=%d, order=%d, %.6f seconds\n",
           SIGNAL_LENGTH, FILTER_ORDER, time_spent);
    
    free(signal);
    free(filtered);
    free(coeffs);
    
    return 0;
}
