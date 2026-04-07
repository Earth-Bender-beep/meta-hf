// Fast inverse square root (Quake III algorithm) and optimizations
// Bitwise manipulation for approximate mathematical operations
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_TESTS 10000000

float fast_inverse_sqrt(float number) {
    long i;
    float x2, y;
    const float threehalfs = 1.5F;
    
    x2 = number * 0.5F;
    y = number;
    i = *(long*)&y;                       // Evil floating point bit hack
    i = 0x5f3759df - (i >> 1);           // What the...?
    y = *(float*)&i;
    y = y * (threehalfs - (x2 * y * y)); // Newton iteration
    
    return y;
}

float regular_inverse_sqrt(float number) {
    return 1.0f / sqrtf(number);
}

// Additional fast math approximations
float fast_sin_approx(float x) {
    // Bhaskara I's sine approximation
    const float PI = 3.14159265f;
    
    // Normalize to [-PI, PI]
    while (x > PI) x -= 2 * PI;
    while (x < -PI) x += 2 * PI;
    
    if (x < 0) {
        return (16.0f * x * (PI + x)) / (5.0f * PI * PI + 4.0f * x * (PI + x));
    } else {
        return (16.0f * x * (PI - x)) / (5.0f * PI * PI + 4.0f * x * (PI - x));
    }
}

float fast_exp_approx(float x) {
    // Fast exponential approximation
    x = 1.0f + x / 256.0f;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x; x *= x; x *= x;
    return x;
}

int main() {
    unsigned int seed = 42;
    
    // Test fast inverse square root
    clock_t start = clock();
    
    double sum_fast = 0.0;
    for (int i = 0; i < NUM_TESTS; i++) {
        seed = seed * 1103515245 + 12345;
        float val = 1.0f + ((seed & 0xFFFF) / (float)0xFFFF) * 100.0f;
        sum_fast += fast_inverse_sqrt(val);
    }
    
    clock_t end = clock();
    double time_fast = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Test regular inverse square root
    seed = 42;
    start = clock();
    
    double sum_regular = 0.0;
    for (int i = 0; i < NUM_TESTS; i++) {
        seed = seed * 1103515245 + 12345;
        float val = 1.0f + ((seed & 0xFFFF) / (float)0xFFFF) * 100.0f;
        sum_regular += regular_inverse_sqrt(val);
    }
    
    end = clock();
    double time_regular = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Test other approximations
    seed = 42;
    start = clock();
    
    double sum_sin = 0.0, sum_exp = 0.0;
    for (int i = 0; i < NUM_TESTS / 10; i++) {
        seed = seed * 1103515245 + 12345;
        float val = ((seed & 0xFFFF) / (float)0xFFFF) * 6.28f;
        sum_sin += fast_sin_approx(val);
        sum_exp += fast_exp_approx(val);
    }
    
    end = clock();
    double time_approx = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Fast math approximations: %d tests\n", NUM_TESTS);
    printf("Fast inv sqrt: %.6f sec, sum=%.4f\n", time_fast, sum_fast);
    printf("Regular inv sqrt: %.6f sec, sum=%.4f, speedup=%.2fx\n", 
           time_regular, sum_regular, time_regular / time_fast);
    printf("Other approximations: %.6f sec\n", time_approx);
    
    return 0;
}
