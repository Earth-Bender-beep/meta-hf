// Autocorrelation function for signal analysis
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SIGNAL_LENGTH 10000
#define MAX_LAG 500

void compute_autocorrelation(double *signal, int length, double *autocorr, int max_lag) {
    double mean = 0.0;
    for (int i = 0; i < length; i++) {
        mean += signal[i];
    }
    mean /= length;
    
    double variance = 0.0;
    for (int i = 0; i < length; i++) {
        double diff = signal[i] - mean;
        variance += diff * diff;
    }
    
    for (int lag = 0; lag < max_lag; lag++) {
        double sum = 0.0;
        int count = 0;
        
        for (int i = 0; i < length - lag; i++) {
            sum += (signal[i] - mean) * (signal[i + lag] - mean);
            count++;
        }
        
        autocorr[lag] = sum / variance;
    }
}

void generate_noisy_sine(double *signal, int length, double frequency, double noise_level, unsigned int *seed) {
    for (int i = 0; i < length; i++) {
        double t = i / 100.0;
        double clean_signal = sin(2.0 * M_PI * frequency * t);
        
        *seed = *seed * 1103515245 + 12345;
        double noise = ((*seed & 0xFFFF) / (double)0xFFFF - 0.5) * 2.0 * noise_level;
        
        signal[i] = clean_signal + noise;
    }
}

int main() {
    int length = SIGNAL_LENGTH;
    int max_lag = MAX_LAG;
    
    double *signal = (double*)malloc(length * sizeof(double));
    double *autocorr = (double*)malloc(max_lag * sizeof(double));
    
    unsigned int seed = 42;
    generate_noisy_sine(signal, length, 0.1, 0.3, &seed);
    
    clock_t start = clock();
    
    compute_autocorrelation(signal, length, autocorr, max_lag);
    
    int max_peak = 0;
    for (int i = 1; i < max_lag; i++) {
        if (autocorr[i] > autocorr[max_peak]) {
            max_peak = i;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Autocorrelation: signal_length=%d, max_lag=%d, %.6f seconds\n",
           length, max_lag, time_spent);
    printf("Max peak at lag: %d, value: %.6f\n", max_peak, autocorr[max_peak]);
    
    free(signal);
    free(autocorr);
    
    return 0;
}
