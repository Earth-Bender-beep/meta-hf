// 1D convolution for signal processing
// Direct convolution algorithm, different from FFT-based methods
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SIGNAL_SIZE 10000
#define KERNEL_SIZE 51

void convolution_1d(double *signal, int sig_len, double *kernel, int ker_len, double *output) {
    int half_kernel = ker_len / 2;
    
    for (int i = 0; i < sig_len; i++) {
        double sum = 0.0;
        
        for (int j = 0; j < ker_len; j++) {
            int sig_idx = i - half_kernel + j;
            
            // Handle boundaries with zero-padding
            if (sig_idx >= 0 && sig_idx < sig_len) {
                sum += signal[sig_idx] * kernel[j];
            }
        }
        
        output[i] = sum;
    }
}

void create_gaussian_kernel(double *kernel, int size, double sigma) {
    int half = size / 2;
    double sum = 0.0;
    
    for (int i = 0; i < size; i++) {
        int x = i - half;
        kernel[i] = exp(-(x * x) / (2.0 * sigma * sigma));
        sum += kernel[i];
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
}

void generate_test_signal(double *signal, int size) {
    for (int i = 0; i < size; i++) {
        // Mix of sine waves
        signal[i] = sin(2.0 * M_PI * 5.0 * i / size) + 
                    0.5 * sin(2.0 * M_PI * 20.0 * i / size) +
                    0.3 * sin(2.0 * M_PI * 50.0 * i / size);
    }
}

int main() {
    double *signal = (double*)malloc(SIGNAL_SIZE * sizeof(double));
    double *kernel = (double*)malloc(KERNEL_SIZE * sizeof(double));
    double *output = (double*)malloc(SIGNAL_SIZE * sizeof(double));
    
    generate_test_signal(signal, SIGNAL_SIZE);
    create_gaussian_kernel(kernel, KERNEL_SIZE, 5.0);
    
    clock_t start = clock();
    convolution_1d(signal, SIGNAL_SIZE, kernel, KERNEL_SIZE, output);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("1D Convolution (signal=%d, kernel=%d): %.6f seconds, output[5000]=%.6f\n",
           SIGNAL_SIZE, KERNEL_SIZE, time_spent, output[5000]);
    
    free(signal);
    free(kernel);
    free(output);
    return 0;
}
