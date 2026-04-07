// Rejection sampling for non-uniform distributions
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_SAMPLES 100000

double uniform_random(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (*seed & 0x7FFFFFFF) / (double)0x7FFFFFFF;
}

double target_distribution(double x) {
    return exp(-x * x / 2.0);
}

double rejection_sample(unsigned int *seed, double (*target)(double), double max_value) {
    while (1) {
        double x = uniform_random(seed) * 4.0 - 2.0;
        double y = uniform_random(seed) * max_value;
        
        if (y <= target(x)) {
            return x;
        }
    }
}

double beta_like_pdf(double x) {
    if (x < 0 || x > 1) return 0;
    return 6 * x * (1 - x);
}

int main() {
    unsigned int seed = 42;
    double *samples = (double*)malloc(NUM_SAMPLES * sizeof(double));
    
    clock_t start = clock();
    
    for (int i = 0; i < NUM_SAMPLES; i++) {
        samples[i] = rejection_sample(&seed, target_distribution, 1.0);
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
        int bin = (int)((samples[i] + 2.0) * bins / 4.0);
        if (bin >= 0 && bin < bins) {
            histogram[bin]++;
        }
    }
    
    int max_bin = 0;
    for (int i = 0; i < bins; i++) {
        if (histogram[i] > max_bin) max_bin = histogram[i];
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Rejection sampling: %d samples, %.6f seconds\n", NUM_SAMPLES, time_spent);
    printf("Mean: %.6f, Variance: %.6f, Max bin: %d\n", mean, variance, max_bin);
    
    free(samples);
    free(histogram);
    
    return 0;
}
