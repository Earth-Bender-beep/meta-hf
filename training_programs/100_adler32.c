// Adler-32 checksum algorithm
// Fast checksum for data integrity verification
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DATA_SIZE 1000000
#define MOD_ADLER 65521

unsigned int adler32(const unsigned char *data, size_t len) {
    unsigned int a = 1, b = 0;
    
    for (size_t i = 0; i < len; i++) {
        a = (a + data[i]) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    
    return (b << 16) | a;
}

// Optimized version with deferred modulo
unsigned int adler32_optimized(const unsigned char *data, size_t len) {
    unsigned int a = 1, b = 0;
    size_t i = 0;
    
    // Process in blocks to defer modulo operations
    while (len > 0) {
        size_t block_len = (len < 5552) ? len : 5552;
        len -= block_len;
        
        for (size_t j = 0; j < block_len; j++) {
            a += data[i++];
            b += a;
        }
        
        a %= MOD_ADLER;
        b %= MOD_ADLER;
    }
    
    return (b << 16) | a;
}

void generate_data(unsigned char *data, size_t size) {
    unsigned int seed = 12345;
    for (size_t i = 0; i < size; i++) {
        seed = seed * 1103515245 + 12345;
        data[i] = seed & 0xFF;
    }
}

int main() {
    unsigned char *data = (unsigned char*)malloc(DATA_SIZE);
    generate_data(data, DATA_SIZE);
    
    clock_t start = clock();
    
    unsigned int checksum = 0;
    for (int iter = 0; iter < 100; iter++) {
        checksum ^= adler32_optimized(data, DATA_SIZE);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Adler-32: %d bytes, 100 iterations, %.6f seconds\n",
           DATA_SIZE, time_spent);
    printf("Final checksum: 0x%08X\n", checksum);
    
    free(data);
    return 0;
}
