// MurmurHash3 32-bit implementation
// Fast non-cryptographic hash function for hash tables
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define NUM_KEYS 100000
#define KEY_LEN 32

uint32_t rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
}

uint32_t murmur3_32(const uint8_t *key, size_t len, uint32_t seed) {
    uint32_t h = seed;
    const int nblocks = len / 4;
    
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;
    
    // Body
    const uint32_t *blocks = (const uint32_t *)(key);
    for (int i = 0; i < nblocks; i++) {
        uint32_t k1 = blocks[i];
        
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;
        
        h ^= k1;
        h = rotl32(h, 13);
        h = h * 5 + 0xe6546b64;
    }
    
    // Tail
    const uint8_t *tail = key + nblocks * 4;
    uint32_t k1 = 0;
    
    switch (len & 3) {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
                k1 *= c1;
                k1 = rotl32(k1, 15);
                k1 *= c2;
                h ^= k1;
    }
    
    // Finalization
    h ^= len;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    
    return h;
}

void generate_key(uint8_t *key, int len, int seed) {
    for (int i = 0; i < len; i++) {
        seed = seed * 1103515245 + 12345;
        key[i] = seed & 0xFF;
    }
}

int main() {
    uint8_t **keys = (uint8_t**)malloc(NUM_KEYS * sizeof(uint8_t*));
    for (int i = 0; i < NUM_KEYS; i++) {
        keys[i] = (uint8_t*)malloc(KEY_LEN);
        generate_key(keys[i], KEY_LEN, i);
    }
    
    clock_t start = clock();
    
    uint32_t hash_sum = 0;
    for (int iter = 0; iter < 10; iter++) {
        for (int i = 0; i < NUM_KEYS; i++) {
            hash_sum ^= murmur3_32(keys[i], KEY_LEN, 42);
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("MurmurHash3: %d keys, 10 iterations, %.6f seconds\n",
           NUM_KEYS, time_spent);
    printf("Hash XOR sum: 0x%08X\n", hash_sum);
    
    for (int i = 0; i < NUM_KEYS; i++) {
        free(keys[i]);
    }
    free(keys);
    
    return 0;
}
