// FNV-1a hash function (32-bit and 64-bit variants)
// Fowler-Noll-Vo hash - simple, fast, good distribution
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define NUM_STRINGS 50000
#define STRING_LEN 64

// FNV-1a 32-bit
uint32_t fnv1a_32(const uint8_t *data, size_t len) {
    uint32_t hash = 2166136261u;  // FNV offset basis
    const uint32_t fnv_prime = 16777619u;
    
    for (size_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    
    return hash;
}

// FNV-1a 64-bit
uint64_t fnv1a_64(const uint8_t *data, size_t len) {
    uint64_t hash = 14695981039346656037ull;  // FNV offset basis
    const uint64_t fnv_prime = 1099511628211ull;
    
    for (size_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    
    return hash;
}

void generate_string(char *str, int len, int seed) {
    for (int i = 0; i < len - 1; i++) {
        seed = seed * 1103515245 + 12345;
        str[i] = 'a' + (seed % 26);
    }
    str[len - 1] = '\0';
}

int main() {
    char **strings = (char**)malloc(NUM_STRINGS * sizeof(char*));
    for (int i = 0; i < NUM_STRINGS; i++) {
        strings[i] = (char*)malloc(STRING_LEN);
        generate_string(strings[i], STRING_LEN, i * 13 + 7);
    }
    
    clock_t start = clock();
    
    uint32_t hash32_sum = 0;
    uint64_t hash64_sum = 0;
    
    for (int iter = 0; iter < 20; iter++) {
        for (int i = 0; i < NUM_STRINGS; i++) {
            hash32_sum ^= fnv1a_32((uint8_t*)strings[i], strlen(strings[i]));
            hash64_sum ^= fnv1a_64((uint8_t*)strings[i], strlen(strings[i]));
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("FNV-1a hash: %d strings, 20 iterations, %.6f seconds\n",
           NUM_STRINGS, time_spent);
    printf("Hash32 XOR: 0x%08X, Hash64 XOR: 0x%016llX\n", 
           hash32_sum, (unsigned long long)hash64_sum);
    
    for (int i = 0; i < NUM_STRINGS; i++) {
        free(strings[i]);
    }
    free(strings);
    
    return 0;
}
