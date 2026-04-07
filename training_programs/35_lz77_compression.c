// LZ77 compression algorithm
// High branching, state machine, string matching
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define WINDOW_SIZE 4096
#define LOOKAHEAD_SIZE 18
#define MIN_MATCH 3

typedef struct {
    int offset;
    int length;
    char next_char;
} Token;

// Find longest match in sliding window
int find_longest_match(const char *data, int pos, int data_len, int *match_offset) {
    int best_length = 0;
    int best_offset = 0;
    
    int window_start = (pos > WINDOW_SIZE) ? pos - WINDOW_SIZE : 0;
    int lookahead_end = (pos + LOOKAHEAD_SIZE < data_len) ? pos + LOOKAHEAD_SIZE : data_len;
    
    for (int i = window_start; i < pos; i++) {
        int match_len = 0;
        while (pos + match_len < lookahead_end && 
               data[i + match_len] == data[pos + match_len]) {
            match_len++;
        }
        
        if (match_len >= MIN_MATCH && match_len > best_length) {
            best_length = match_len;
            best_offset = pos - i;
        }
    }
    
    *match_offset = best_offset;
    return best_length;
}

int lz77_compress(const char *input, int input_len, Token *output) {
    int pos = 0;
    int token_count = 0;
    
    while (pos < input_len) {
        int match_offset = 0;
        int match_length = find_longest_match(input, pos, input_len, &match_offset);
        
        Token token;
        if (match_length >= MIN_MATCH) {
            token.offset = match_offset;
            token.length = match_length;
            token.next_char = (pos + match_length < input_len) ? 
                             input[pos + match_length] : '\0';
            pos += match_length + 1;
        } else {
            token.offset = 0;
            token.length = 0;
            token.next_char = input[pos];
            pos++;
        }
        
        output[token_count++] = token;
    }
    
    return token_count;
}

void generate_test_data(char *data, int size) {
    const char *pattern = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    int pattern_len = strlen(pattern);
    
    for (int i = 0; i < size; i++) {
        if (i < size / 2) {
            data[i] = pattern[i % pattern_len];
        } else {
            // Repeated data for compression
            data[i] = pattern[(i / 4) % pattern_len];
        }
    }
}

int main() {
    int input_size = 8192;
    char *input = (char*)malloc(input_size);
    Token *output = (Token*)malloc(input_size * sizeof(Token));
    
    generate_test_data(input, input_size);
    
    clock_t start = clock();
    int compressed_tokens = lz77_compress(input, input_size, output);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    double ratio = (double)compressed_tokens * sizeof(Token) / input_size;
    
    printf("LZ77 compression: %.6f seconds, %d->%d tokens, ratio=%.2f\n",
           time_spent, input_size, compressed_tokens, ratio);
    
    free(input);
    free(output);
    return 0;
}
