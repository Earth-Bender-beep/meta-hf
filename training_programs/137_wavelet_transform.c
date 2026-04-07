// Haar wavelet transform for signal processing and compression
// Simple discrete wavelet transform implementation
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SIGNAL_SIZE 8192
#define NUM_TRANSFORMS 500

void haar_transform_1d(double *signal, int n) {
    double *temp = (double*)malloc(n * sizeof(double));
    
    while (n > 1) {
        int half = n / 2;
        
        // Compute averages and differences
        for (int i = 0; i < half; i++) {
            temp[i] = (signal[2*i] + signal[2*i + 1]) / 2.0;
            temp[half + i] = (signal[2*i] - signal[2*i + 1]) / 2.0;
        }
        
        // Copy back
        for (int i = 0; i < n; i++) {
            signal[i] = temp[i];
        }
        
        n = half;
    }
    
    free(temp);
}

void inverse_haar_transform_1d(double *signal, int original_size) {
    double *temp = (double*)malloc(original_size * sizeof(double));
    
    int n = 1;
    while (n < original_size) {
        int half = n;
        n *= 2;
        
        // Reconstruct from averages and differences
        for (int i = 0; i < half; i++) {
            temp[2*i] = signal[i] + signal[half + i];
            temp[2*i + 1] = signal[i] - signal[half + i];
        }
        
        // Copy back
        for (int i = 0; i < n; i++) {
            signal[i] = temp[i];
        }
    }
    
    free(temp);
}

void haar_transform_2d(double **image, int width, int height) {
    // Transform rows
    for (int i = 0; i < height; i++) {
        haar_transform_1d(image[i], width);
    }
    
    // Transform columns
    double *col = (double*)malloc(height * sizeof(double));
    for (int j = 0; j < width; j++) {
        for (int i = 0; i < height; i++) {
            col[i] = image[i][j];
        }
        haar_transform_1d(col, height);
        for (int i = 0; i < height; i++) {
            image[i][j] = col[i];
        }
    }
    free(col);
}

double compress_signal(double *signal, int n, double threshold) {
    // Zero out small coefficients
    int removed = 0;
    for (int i = 0; i < n; i++) {
        if (fabs(signal[i]) < threshold) {
            signal[i] = 0.0;
            removed++;
        }
    }
    return removed / (double)n;
}

void generate_signal(double *signal, int n) {
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        signal[i] = sin(2.0 * 3.14159 * i / 100.0) + 
                    0.5 * sin(2.0 * 3.14159 * i / 20.0) +
                    0.1 * ((seed & 0xFFFF) / (double)0xFFFF - 0.5);
    }
}

double compute_mse(double *signal1, double *signal2, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = signal1[i] - signal2[i];
        sum += diff * diff;
    }
    return sum / n;
}

int main() {
    double *signal = (double*)malloc(SIGNAL_SIZE * sizeof(double));
    double *original = (double*)malloc(SIGNAL_SIZE * sizeof(double));
    double *compressed = (double*)malloc(SIGNAL_SIZE * sizeof(double));
    
    generate_signal(original, SIGNAL_SIZE);
    
    clock_t start = clock();
    
    double total_compression = 0.0;
    double total_mse = 0.0;
    
    for (int iter = 0; iter < NUM_TRANSFORMS; iter++) {
        // Copy original
        for (int i = 0; i < SIGNAL_SIZE; i++) {
            signal[i] = original[i];
        }
        
        // Forward transform
        haar_transform_1d(signal, SIGNAL_SIZE);
        
        // Compress
        for (int i = 0; i < SIGNAL_SIZE; i++) {
            compressed[i] = signal[i];
        }
        double ratio = compress_signal(compressed, SIGNAL_SIZE, 0.1);
        total_compression += ratio;
        
        // Inverse transform
        inverse_haar_transform_1d(compressed, SIGNAL_SIZE);
        
        // Compute error
        double mse = compute_mse(original, compressed, SIGNAL_SIZE);
        total_mse += mse;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Haar wavelet transform: signal=%d, %d transforms, %.6f seconds\n",
           SIGNAL_SIZE, NUM_TRANSFORMS, time_spent);
    printf("Average compression: %.2f%%, MSE: %.6f\n",
           total_compression / NUM_TRANSFORMS * 100, total_mse / NUM_TRANSFORMS);
    
    free(signal);
    free(original);
    free(compressed);
    
    return 0;
}
