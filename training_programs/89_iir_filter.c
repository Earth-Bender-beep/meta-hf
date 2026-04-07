// IIR (Infinite Impulse Response) recursive digital filter
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SIGNAL_LENGTH 20000
#define FILTER_ORDER 4

typedef struct {
    double b[FILTER_ORDER + 1];  // Feedforward coefficients
    double a[FILTER_ORDER + 1];  // Feedback coefficients
    int order;
} IIRFilter;

void design_butterworth_lowpass(IIRFilter *filter, double cutoff) {
    filter->order = FILTER_ORDER;
    
    // Simplified 2nd order Butterworth coefficients
    double omega = tan(M_PI * cutoff);
    double omega2 = omega * omega;
    double sqrt2 = sqrt(2.0);
    
    double k = omega2 / (1.0 + sqrt2 * omega + omega2);
    
    filter->b[0] = k;
    filter->b[1] = 2.0 * k;
    filter->b[2] = k;
    
    filter->a[0] = 1.0;
    filter->a[1] = 2.0 * k - 2.0 * (omega2 - 1.0) / (1.0 + sqrt2 * omega + omega2);
    filter->a[2] = k - sqrt2 * omega / (1.0 + sqrt2 * omega + omega2) + 1.0;
}

void iir_filter(double *input, double *output, IIRFilter *filter, int length) {
    double x[FILTER_ORDER + 1] = {0};
    double y[FILTER_ORDER + 1] = {0};
    
    for (int n = 0; n < length; n++) {
        // Shift input history
        for (int i = FILTER_ORDER; i > 0; i--) {
            x[i] = x[i - 1];
        }
        x[0] = input[n];
        
        // Compute output
        output[n] = 0.0;
        for (int i = 0; i <= 2; i++) {
            output[n] += filter->b[i] * x[i];
        }
        for (int i = 1; i <= 2; i++) {
            output[n] -= filter->a[i] * y[i];
        }
        
        // Shift output history
        for (int i = FILTER_ORDER; i > 0; i--) {
            y[i] = y[i - 1];
        }
        y[0] = output[n];
    }
}

int main() {
    double *signal = (double*)malloc(SIGNAL_LENGTH * sizeof(double));
    double *filtered = (double*)malloc(SIGNAL_LENGTH * sizeof(double));
    IIRFilter filter;
    
    unsigned int seed = 42;
    for (int i = 0; i < SIGNAL_LENGTH; i++) {
        seed = seed * 1103515245 + 12345;
        signal[i] = sin(2.0 * M_PI * 0.01 * i) + 
                    0.5 * sin(2.0 * M_PI * 0.3 * i) +
                    0.2 * ((seed & 0xFFFF) / (double)0xFFFF - 0.5);
    }
    
    design_butterworth_lowpass(&filter, 0.1);
    
    clock_t start = clock();
    iir_filter(signal, filtered, &filter, SIGNAL_LENGTH);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("IIR filter: signal=%d, order=%d, %.6f seconds\n",
           SIGNAL_LENGTH, FILTER_ORDER, time_spent);
    
    free(signal);
    free(filtered);
    
    return 0;
}
