// Base64 encoding - binary to text encoding
// Table lookups, bit manipulation, padding
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DATA_SIZE 5000

static const char base64_table[] = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int base64_encode(const unsigned char *input, int input_len, char *output) {
    int output_idx = 0;
    
    for (int i = 0; i < input_len; i += 3) {
        unsigned int triple = 0;
        int bytes_remaining = input_len - i;
        
        // Build 24-bit triple from 3 bytes
        triple = input[i] << 16;
        if (bytes_remaining > 1) triple |= input[i+1] << 8;
        if (bytes_remaining > 2) triple |= input[i+2];
        
        // Extract four 6-bit groups
        output[output_idx++] = base64_table[(triple >> 18) & 0x3F];
        output[output_idx++] = base64_table[(triple >> 12) & 0x3F];
        output[output_idx++] = (bytes_remaining > 1) ? base64_table[(triple >> 6) & 0x3F] : '=';
        output[output_idx++] = (bytes_remaining > 2) ? base64_table[triple & 0x3F] : '=';
    }
    
    output[output_idx] = '\0';
    return output_idx;
}

int base64_decode(const char *input, int input_len, unsigned char *output) {
    // Decode table
    int decode_table[256];
    for (int i = 0; i < 256; i++) decode_table[i] = -1;
    for (int i = 0; i < 64; i++) decode_table[(int)base64_table[i]] = i;
    
    int output_idx = 0;
    
    for (int i = 0; i < input_len; i += 4) {
        if (input[i] == '=' || input[i+1] == '=') break;
        
        unsigned int quad = 0;
        quad = decode_table[(int)input[i]] << 18;
        quad |= decode_table[(int)input[i+1]] << 12;
        
        output[output_idx++] = (quad >> 16) & 0xFF;
        
        if (input[i+2] != '=') {
            quad |= decode_table[(int)input[i+2]] << 6;
            output[output_idx++] = (quad >> 8) & 0xFF;
        }
        
        if (input[i+3] != '=') {
            quad |= decode_table[(int)input[i+3]];
            output[output_idx++] = quad & 0xFF;
        }
    }
    
    return output_idx;
}

void generate_binary_data(unsigned char *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (unsigned char)((i * 17 + 23) % 256);
    }
}

int main() {
    unsigned char *input = (unsigned char*)malloc(DATA_SIZE);
    char *encoded = (char*)malloc((DATA_SIZE * 4 / 3 + 4) * sizeof(char));
    unsigned char *decoded = (unsigned char*)malloc(DATA_SIZE);
    
    generate_binary_data(input, DATA_SIZE);
    
    clock_t start = clock();
    int encoded_len = base64_encode(input, DATA_SIZE, encoded);
    int decoded_len = base64_decode(encoded, encoded_len, decoded);
    clock_t end = clock();
    
    // Verify correctness
    int errors = 0;
    for (int i = 0; i < DATA_SIZE; i++) {
        if (input[i] != decoded[i]) errors++;
    }
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Base64 encode/decode: %d bytes -> %d chars -> %d bytes, %.6f seconds\n",
           DATA_SIZE, encoded_len, decoded_len, time_spent);
    printf("Errors: %d, Expansion: %.1f%%\n", errors, 
           100.0 * (encoded_len - DATA_SIZE) / DATA_SIZE);
    
    free(input);
    free(encoded);
    free(decoded);
    return 0;
}
