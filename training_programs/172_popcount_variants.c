// Various bit counting methods and population count algorithms
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_TESTS 1000000

int popcount_naive(unsigned int x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

int popcount_brian_kernighan(unsigned int x) {
    int count = 0;
    while (x) {
        x &= x - 1;
        count++;
    }
    return count;
}

int popcount_lookup(unsigned int x) {
    static int table[256] = {0};
    static int initialized = 0;
    
    if (!initialized) {
        for (int i = 0; i < 256; i++) {
            table[i] = (i & 1) + table[i / 2];
        }
        initialized = 1;
    }
    
    return table[x & 0xFF] + table[(x >> 8) & 0xFF] +
           table[(x >> 16) & 0xFF] + table[(x >> 24) & 0xFF];
}

int popcount_parallel(unsigned int x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    x = x + (x >> 8);
    x = x + (x >> 16);
    return x & 0x3F;
}

int parity(unsigned int x) {
    x ^= x >> 16;
    x ^= x >> 8;
    x ^= x >> 4;
    x &= 0xF;
    return (0x6996 >> x) & 1;
}

int main() {
    unsigned int *test_data = (unsigned int*)malloc(NUM_TESTS * sizeof(unsigned int));
    
    unsigned int seed = 42;
    for (int i = 0; i < NUM_TESTS; i++) {
        seed = seed * 1103515245 + 12345;
        test_data[i] = seed;
    }
    
    clock_t start = clock();
    
    long long sum_naive = 0;
    for (int i = 0; i < NUM_TESTS; i++) {
        sum_naive += popcount_naive(test_data[i]);
    }
    
    long long sum_bk = 0;
    for (int i = 0; i < NUM_TESTS; i++) {
        sum_bk += popcount_brian_kernighan(test_data[i]);
    }
    
    long long sum_lookup = 0;
    for (int i = 0; i < NUM_TESTS; i++) {
        sum_lookup += popcount_lookup(test_data[i]);
    }
    
    long long sum_parallel = 0;
    for (int i = 0; i < NUM_TESTS; i++) {
        sum_parallel += popcount_parallel(test_data[i]);
    }
    
    long long sum_parity = 0;
    for (int i = 0; i < NUM_TESTS; i++) {
        sum_parity += parity(test_data[i]);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Popcount variants: %d tests, %.6f seconds\n", NUM_TESTS, time_spent);
    printf("Results: naive=%lld, bk=%lld, lookup=%lld, parallel=%lld, parity=%lld\n",
           sum_naive, sum_bk, sum_lookup, sum_parallel, sum_parity);
    
    free(test_data);
    
    return 0;
}
