// Euclidean algorithm variants for GCD computation
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_TESTS 1000000

int gcd_recursive(int a, int b) {
    if (b == 0) return a;
    return gcd_recursive(b, a % b);
}

int gcd_iterative(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

int gcd_binary(int a, int b) {
    if (a == 0) return b;
    if (b == 0) return a;
    
    int shift = 0;
    while (((a | b) & 1) == 0) {
        a >>= 1;
        b >>= 1;
        shift++;
    }
    
    while ((a & 1) == 0) a >>= 1;
    
    do {
        while ((b & 1) == 0) b >>= 1;
        
        if (a > b) {
            int temp = a;
            a = b;
            b = temp;
        }
        
        b = b - a;
    } while (b != 0);
    
    return a << shift;
}

long long extended_gcd(long long a, long long b, long long *x, long long *y) {
    if (b == 0) {
        *x = 1;
        *y = 0;
        return a;
    }
    
    long long x1, y1;
    long long gcd = extended_gcd(b, a % b, &x1, &y1);
    
    *x = y1;
    *y = x1 - (a / b) * y1;
    
    return gcd;
}

int main() {
    unsigned int seed = 42;
    int *test_a = (int*)malloc(NUM_TESTS * sizeof(int));
    int *test_b = (int*)malloc(NUM_TESTS * sizeof(int));
    
    for (int i = 0; i < NUM_TESTS; i++) {
        seed = seed * 1103515245 + 12345;
        test_a[i] = (seed % 10000) + 1;
        seed = seed * 1103515245 + 12345;
        test_b[i] = (seed % 10000) + 1;
    }
    
    clock_t start = clock();
    
    long long sum_recursive = 0;
    for (int i = 0; i < NUM_TESTS; i++) {
        sum_recursive += gcd_recursive(test_a[i], test_b[i]);
    }
    
    long long sum_iterative = 0;
    for (int i = 0; i < NUM_TESTS; i++) {
        sum_iterative += gcd_iterative(test_a[i], test_b[i]);
    }
    
    long long sum_binary = 0;
    for (int i = 0; i < NUM_TESTS; i++) {
        sum_binary += gcd_binary(test_a[i], test_b[i]);
    }
    
    long long sum_extended = 0;
    for (int i = 0; i < 1000; i++) {
        long long x, y;
        sum_extended += extended_gcd(test_a[i], test_b[i], &x, &y);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("GCD algorithms: %d tests, %.6f seconds\n", NUM_TESTS, time_spent);
    printf("Sums: recursive=%lld, iterative=%lld, binary=%lld\n",
           sum_recursive, sum_iterative, sum_binary);
    
    free(test_a);
    free(test_b);
    
    return 0;
}
