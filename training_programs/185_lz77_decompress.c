// LZ77 decompression algorithm
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define WINDOW_SIZE 4096
#define LOOKAHEAD_SIZE 18

typedef struct {
    int offset;
    int length;
    char next_char;
} Token;

void lz77_decompress(Token *tokens, int num_tokens, char *output, int *output_len) {
    *output_len = 0;
    
    for (int i = 0; i < num_tokens; i++) {
        if (tokens[i].length > 0) {
            int start = *output_len - tokens[i].offset;
            for (int j = 0; j < tokens[i].length; j++) {
                output[(*output_len)++] = output[start + j];
            }
        }
        
        if (tokens[i].next_char != '\0') {
            output[(*output_len)++] = tokens[i].next_char;
        }
    }
    
    output[*output_len] = '\0';
}

int main() {
    Token tokens[] = {
        {0, 0, 'a'},
        {0, 0, 'b'},
        {0, 0, 'c'},
        {3, 3, 'd'},
        {6, 6, 'e'}
    };
    
    int num_tokens = 5;
    char *output = (char*)malloc(10000);
    
    clock_t start = clock();
    
    int total_len = 0;
    for (int test = 0; test < 100000; test++) {
        int output_len;
        lz77_decompress(tokens, num_tokens, output, &output_len);
        total_len += output_len;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("LZ77 decompress: 100000 iterations, %.6f seconds\n", time_spent);
    printf("Total decompressed length: %d\n", total_len);
    
    free(output);
    return 0;
}
