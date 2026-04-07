// Bit reversal algorithms for FFT and other applications
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_TESTS 1000000

unsigned int reverse_bits_naive(unsigned int x) {
    unsigned int result = 0;
    for (int i = 0; i < 32; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

unsigned int reverse_bits_lookup(unsigned int x) {
    static unsigned char table[256] = {0};
    static int initialized = 0;
    
    if (!initialized) {
        for (int i = 0; i < 256; i++) {
            unsigned char val = 0;
            for (int j = 0; j < 8; j++) {
                val = (val << 1) | ((i >> j) & 1);
            }
            table[i] = val;
        }
        initialized = 1;
    }
    
    return (table[x & 0xFF] << 24) |
           (table[(x >> 8) & 0xFF] << 16) |
           (table[(x >> 16) & 0xFF] << 8) |
           (table[(x >> 24) & 0xFF]);
}

unsigned int reverse_bits_parallel(unsigned int x) {
    x = ((x & 0x55555555) << 1) | ((x >> 1) & 0x55555555);
    x = ((x & 0x33333333) << 2) | ((x >> 2) & 0x33333333);
    x = ((x & 0x0F0F0F0F) << 4) | ((x >> 4) & 0x0F0F0F0F);
    x = ((x & 0x00FF00FF) << 8) | ((x >> 8) & 0x00FF00FF);
    x = (x << 16) | (x >> 16);
    return x;
}

void bit_reverse_permutation(unsigned int *array, int n) {
    int bits = 0;
    int temp = n - 1;
    while (temp > 0) {
        bits++;
        temp >>= 1;
    }
    
    for (int i = 0; i < n; i++) {
        int j = 0;
        int temp_i = i;
        for (int b = 0; b < bits; b++) {
            j = (j << 1) | (temp_i & 1);
            temp_i >>= 1;
        }
        
        if (j > i) {
            unsigned int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
}

int main() {
    unsigned int *test_data = (unsigned int*)malloc(NUM_TESTS * sizeof(unsigned int));
    unsigned int *results = (unsigned int*)malloc(NUM_TESTS * sizeof(unsigned int));
    
    unsigned int seed = 42;
    for (int i = 0; i < NUM_TESTS; i++) {
        seed = seed * 1103515245 + 12345;
        test_data[i] = seed;
    }
    
    clock_t start = clock();
    
    for (int i = 0; i < NUM_TESTS; i++) {
        results[i] = reverse_bits_naive(test_data[i]);
    }
    
    for (int i = 0; i < NUM_TESTS; i++) {
        results[i] = reverse_bits_lookup(test_data[i]);
    }
    
    for (int i = 0; i < NUM_TESTS; i++) {
        results[i] = reverse_bits_parallel(test_data[i]);
    }
    
    unsigned int array[1024];
    for (int i = 0; i < 1024; i++) {
        array[i] = i;
    }
    bit_reverse_permutation(array, 1024);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Bit reversal: %d operations, %.6f seconds\n", NUM_TESTS * 3, time_spent);
    printf("Sample result: 0x%08X reversed = 0x%08X\n", test_data[0], results[0]);
    
    free(test_data);
    free(results);
    
    return 0;
}
