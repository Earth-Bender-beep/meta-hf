// UTF-8 sequence validator and character counter
// Validates multi-byte UTF-8 encoding rules
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TEXT_SIZE 500000

int utf8_char_length(unsigned char byte) {
    if ((byte & 0x80) == 0) return 1;        // 0xxxxxxx
    if ((byte & 0xE0) == 0xC0) return 2;     // 110xxxxx
    if ((byte & 0xF0) == 0xE0) return 3;     // 1110xxxx
    if ((byte & 0xF8) == 0xF0) return 4;     // 11110xxx
    return -1; // invalid
}

int is_continuation_byte(unsigned char byte) {
    return (byte & 0xC0) == 0x80;  // 10xxxxxx
}

int validate_utf8(const unsigned char *text, int len, int *char_count) {
    int i = 0;
    int chars = 0;
    
    while (i < len) {
        int char_len = utf8_char_length(text[i]);
        
        if (char_len < 0) return 0; // invalid start byte
        
        if (i + char_len > len) return 0; // truncated sequence
        
        // Check continuation bytes
        for (int j = 1; j < char_len; j++) {
            if (!is_continuation_byte(text[i + j])) {
                return 0; // invalid continuation
            }
        }
        
        // Check for overlong encoding (simplified check)
        if (char_len == 2) {
            unsigned int codepoint = ((text[i] & 0x1F) << 6) | (text[i+1] & 0x3F);
            if (codepoint < 0x80) return 0; // overlong
        } else if (char_len == 3) {
            unsigned int codepoint = ((text[i] & 0x0F) << 12) | 
                                    ((text[i+1] & 0x3F) << 6) | 
                                    (text[i+2] & 0x3F);
            if (codepoint < 0x800) return 0; // overlong
            // Check for surrogates
            if (codepoint >= 0xD800 && codepoint <= 0xDFFF) return 0;
        } else if (char_len == 4) {
            unsigned int codepoint = ((text[i] & 0x07) << 18) | 
                                    ((text[i+1] & 0x3F) << 12) | 
                                    ((text[i+2] & 0x3F) << 6) | 
                                    (text[i+3] & 0x3F);
            if (codepoint < 0x10000 || codepoint > 0x10FFFF) return 0;
        }
        
        i += char_len;
        chars++;
    }
    
    *char_count = chars;
    return 1;
}

void generate_utf8_text(unsigned char *text, int size) {
    int pos = 0;
    unsigned int seed = 42;
    
    while (pos < size - 4) {
        seed = seed * 1103515245 + 12345;
        int type = seed % 100;
        
        if (type < 70) {
            // 1-byte ASCII
            text[pos++] = 0x20 + (seed % 95);
        } else if (type < 90) {
            // 2-byte character (U+0080 to U+07FF)
            unsigned int codepoint = 0x80 + (seed % 0x780);
            text[pos++] = 0xC0 | (codepoint >> 6);
            text[pos++] = 0x80 | (codepoint & 0x3F);
        } else if (type < 98) {
            // 3-byte character (U+0800 to U+FFFF, excluding surrogates)
            unsigned int codepoint = 0x800 + (seed % 0xD800);
            text[pos++] = 0xE0 | (codepoint >> 12);
            text[pos++] = 0x80 | ((codepoint >> 6) & 0x3F);
            text[pos++] = 0x80 | (codepoint & 0x3F);
        } else {
            // 4-byte character (U+10000 to U+10FFFF)
            unsigned int codepoint = 0x10000 + (seed % 0xFFFF);
            text[pos++] = 0xF0 | (codepoint >> 18);
            text[pos++] = 0x80 | ((codepoint >> 12) & 0x3F);
            text[pos++] = 0x80 | ((codepoint >> 6) & 0x3F);
            text[pos++] = 0x80 | (codepoint & 0x3F);
        }
    }
    text[pos] = '\0';
}

int main() {
    unsigned char *text = (unsigned char*)malloc(TEXT_SIZE);
    generate_utf8_text(text, TEXT_SIZE);
    int actual_len = strlen((char*)text);
    
    clock_t start = clock();
    
    int total_chars = 0;
    int valid_count = 0;
    
    for (int iter = 0; iter < 100; iter++) {
        int char_count = 0;
        if (validate_utf8(text, actual_len, &char_count)) {
            valid_count++;
            total_chars += char_count;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("UTF-8 validator: %d bytes, 100 iterations, %.6f seconds\n",
           actual_len, time_spent);
    printf("Valid: %d/100, Avg chars: %d\n", valid_count, total_chars / 100);
    
    free(text);
    return 0;
}
