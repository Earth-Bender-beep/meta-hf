// Box-Muller transform for Gaussian random numbers
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_SAMPLES 500000
#define M_PI 3.14159265358979323846

void box_muller(double u1, double u2, double *z0, double *z1) {
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    
    *z0 = r * cos(theta);
    *z1 = r * sin(theta);
}

double uniform_random(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (*seed & 0x7FFFFFFF) / (double)0x7FFFFFFF;
}

int main() {
    double *samples = (double*)malloc(NUM_SAMPLES * sizeof(double));
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    for (int i = 0; i < NUM_SAMPLES; i += 2) {
        double u1 = uniform_random(&seed);
        double u2 = uniform_random(&seed);
        
        double z0, z1;
        box_muller(u1, u2, &z0, &z1);
        
        samples[i] = z0;
        if (i + 1 < NUM_SAMPLES) {
            samples[i + 1] = z1;
        }
    }
    
    double mean = 0.0;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        mean += samples[i];
    }
    mean /= NUM_SAMPLES;
    
    double variance = 0.0;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        double diff = samples[i] - mean;
        variance += diff * diff;
    }
    variance /= NUM_SAMPLES;
    
    int bins = 50;
    int *histogram = (int*)calloc(bins, sizeof(int));
    
    for (int i = 0; i < NUM_SAMPLES; i++) {
        int bin = (int)((samples[i] + 4.0) * bins / 8.0);
        if (bin >= 0 && bin < bins) {
            histogram[bin]++;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Box-Muller: %d samples, %.6f seconds\n", NUM_SAMPLES, time_spent);
    printf("Mean: %.6f, Variance: %.6f\n", mean, variance);
    
    free(samples);
    free(histogram);
    
    return 0;
}
