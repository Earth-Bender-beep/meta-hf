// Linear congruential generator and random number tests
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_SAMPLES 1000000

typedef struct {
    unsigned long long a, c, m;
    unsigned long long seed;
} LCG;

void init_lcg(LCG *lcg, unsigned long long seed, unsigned long long a, unsigned long long c, unsigned long long m) {
    lcg->seed = seed;
    lcg->a = a;
    lcg->c = c;
    lcg->m = m;
}

unsigned long long next_lcg(LCG *lcg) {
    lcg->seed = (lcg->a * lcg->seed + lcg->c) % lcg->m;
    return lcg->seed;
}

double uniform_lcg(LCG *lcg) {
    return next_lcg(lcg) / (double)lcg->m;
}

double chi_square_test(unsigned int *observed, int bins, int expected) {
    double chi2 = 0.0;
    for (int i = 0; i < bins; i++) {
        double diff = observed[i] - expected;
        chi2 += (diff * diff) / expected;
    }
    return chi2;
}

int main() {
    LCG lcg;
    init_lcg(&lcg, 42, 1103515245, 12345, 2147483648ULL);
    
    int bins = 100;
    unsigned int *histogram = (unsigned int*)calloc(bins, sizeof(unsigned int));
    
    clock_t start = clock();
    
    for (int i = 0; i < NUM_SAMPLES; i++) {
        double u = uniform_lcg(&lcg);
        int bin = (int)(u * bins);
        if (bin >= bins) bin = bins - 1;
        histogram[bin]++;
    }
    
    double chi2 = chi_square_test(histogram, bins, NUM_SAMPLES / bins);
    
    double mean = 0.0;
    double variance = 0.0;
    
    init_lcg(&lcg, 42, 1103515245, 12345, 2147483648ULL);
    for (int i = 0; i < 10000; i++) {
        mean += uniform_lcg(&lcg);
    }
    mean /= 10000;
    
    init_lcg(&lcg, 42, 1103515245, 12345, 2147483648ULL);
    for (int i = 0; i < 10000; i++) {
        double x = uniform_lcg(&lcg);
        variance += (x - mean) * (x - mean);
    }
    variance /= 10000;
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("LCG: %d samples, %.6f seconds\n", NUM_SAMPLES, time_spent);
    printf("Chi-square: %.2f, Mean: %.6f, Variance: %.6f\n", chi2, mean, variance);
    
    free(histogram);
    return 0;
}
