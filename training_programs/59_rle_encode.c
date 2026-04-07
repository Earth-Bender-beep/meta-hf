// Run-Length Encoding (RLE) compression
// Simple compression for repetitive data
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DATA_SIZE 10000

typedef struct {
    unsigned char value;
    int count;
} RLE_Pair;

int rle_encode(unsigned char *input, int input_len, RLE_Pair *output) {
    if (input_len == 0) return 0;
    
    int output_idx = 0;
    int i = 0;
    
    while (i < input_len) {
        unsigned char current = input[i];
        int count = 1;
        
        // Count consecutive identical bytes
        while (i + count < input_len && input[i + count] == current && count < 255) {
            count++;
        }
        
        output[output_idx].value = current;
        output[output_idx].count = count;
        output_idx++;
        
        i += count;
    }
    
    return output_idx;
}

int rle_decode(RLE_Pair *input, int input_len, unsigned char *output) {
    int output_idx = 0;
    
    for (int i = 0; i < input_len; i++) {
        for (int j = 0; j < input[i].count; j++) {
            output[output_idx++] = input[i].value;
        }
    }
    
    return output_idx;
}

void generate_repetitive_data(unsigned char *data, int size) {
    int pos = 0;
    
    while (pos < size) {
        unsigned char value = (unsigned char)(pos % 256);
        int run_length = 10 + (pos % 50);
        
        for (int i = 0; i < run_length && pos < size; i++) {
            data[pos++] = value;
        }
    }
}

int main() {
    unsigned char *input = (unsigned char*)malloc(DATA_SIZE);
    RLE_Pair *encoded = (RLE_Pair*)malloc(DATA_SIZE * sizeof(RLE_Pair));
    unsigned char *decoded = (unsigned char*)malloc(DATA_SIZE);
    
    generate_repetitive_data(input, DATA_SIZE);
    
    clock_t start = clock();
    int encoded_len = rle_encode(input, DATA_SIZE, encoded);
    int decoded_len = rle_decode(encoded, encoded_len, decoded);
    clock_t end = clock();
    
    // Verify correctness
    int errors = 0;
    for (int i = 0; i < DATA_SIZE; i++) {
        if (input[i] != decoded[i]) errors++;
    }
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    double compression_ratio = (double)(encoded_len * sizeof(RLE_Pair)) / DATA_SIZE;
    
    printf("RLE encoding/decoding: %d bytes -> %d pairs, %.6f seconds\n",
           DATA_SIZE, encoded_len, time_spent);
    printf("Compression ratio: %.2f%%, errors: %d\n", compression_ratio * 100, errors);
    
    free(input);
    free(encoded);
    free(decoded);
    return 0;
}
