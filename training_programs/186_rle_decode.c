// Run-length encoding decoder
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_SIZE 100000

int rle_decode(unsigned char *encoded, int encoded_len, unsigned char *decoded) {
    int decoded_len = 0;
    
    for (int i = 0; i < encoded_len; i += 2) {
        unsigned char count = encoded[i];
        unsigned char value = encoded[i + 1];
        
        for (int j = 0; j < count; j++) {
            decoded[decoded_len++] = value;
        }
    }
    
    return decoded_len;
}

int rle_encode(unsigned char *input, int input_len, unsigned char *encoded) {
    int encoded_len = 0;
    int i = 0;
    
    while (i < input_len) {
        unsigned char current = input[i];
        int count = 1;
        
        while (i + count < input_len && input[i + count] == current && count < 255) {
            count++;
        }
        
        encoded[encoded_len++] = count;
        encoded[encoded_len++] = current;
        i += count;
    }
    
    return encoded_len;
}

int main() {
    unsigned char *input = (unsigned char*)malloc(MAX_SIZE);
    unsigned char *encoded = (unsigned char*)malloc(MAX_SIZE * 2);
    unsigned char *decoded = (unsigned char*)malloc(MAX_SIZE);
    
    unsigned int seed = 42;
    for (int i = 0; i < 1000; i++) {
        seed = seed * 1103515245 + 12345;
        unsigned char val = seed & 0x0F;
        int run = (seed >> 8) % 10 + 1;
        for (int j = 0; j < run && i < 1000; j++) {
            input[i++] = val;
        }
    }
    
    clock_t start = clock();
    
    long long total_decoded = 0;
    for (int test = 0; test < 10000; test++) {
        int encoded_len = rle_encode(input, 1000, encoded);
        int decoded_len = rle_decode(encoded, encoded_len, decoded);
        total_decoded += decoded_len;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("RLE decode: 10000 iterations, %.6f seconds\n", time_spent);
    printf("Total decoded: %lld bytes\n", total_decoded);
    
    free(input);
    free(encoded);
    free(decoded);
    
    return 0;
}
