// Bootstrap resampling for statistical estimation
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define DATA_SIZE 1000
#define NUM_BOOTSTRAPS 500

double compute_mean(int *data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum / n;
}

double compute_variance(int *data, int n, double mean) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = data[i] - mean;
        sum += diff * diff;
    }
    return sum / n;
}

void bootstrap_sample(int *original, int *sample, int n, unsigned int *seed) {
    for (int i = 0; i < n; i++) {
        *seed = *seed * 1103515245 + 12345;
        int idx = (*seed) % n;
        sample[i] = original[idx];
    }
}

int main() {
    int *data = (int*)malloc(DATA_SIZE * sizeof(int));
    int *sample = (int*)malloc(DATA_SIZE * sizeof(int));
    double *bootstrap_means = (double*)malloc(NUM_BOOTSTRAPS * sizeof(double));
    
    unsigned int seed = 42;
    for (int i = 0; i < DATA_SIZE; i++) {
        seed = seed * 1103515245 + 12345;
        data[i] = seed % 1000;
    }
    
    clock_t start = clock();
    
    for (int b = 0; b < NUM_BOOTSTRAPS; b++) {
        bootstrap_sample(data, sample, DATA_SIZE, &seed);
        bootstrap_means[b] = compute_mean(sample, DATA_SIZE);
    }
    
    double mean_of_means = compute_mean((int*)bootstrap_means, NUM_BOOTSTRAPS);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Bootstrap: %d samples, %d bootstraps, %.6f seconds\n",
           DATA_SIZE, NUM_BOOTSTRAPS, time_spent);
    
    free(data);
    free(sample);
    free(bootstrap_means);
    
    return 0;
}
