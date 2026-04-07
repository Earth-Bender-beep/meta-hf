// Z-algorithm for linear-time pattern matching
// Computes Z-array: length of longest substring starting from each position that matches prefix
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TEXT_SIZE 500000
#define PATTERN_SIZE 100

void compute_z_array(const char *str, int n, int *z) {
    int l = 0, r = 0;
    z[0] = n;
    
    for (int i = 1; i < n; i++) {
        if (i > r) {
            l = r = i;
            while (r < n && str[r - l] == str[r]) {
                r++;
            }
            z[i] = r - l;
            r--;
        } else {
            int k = i - l;
            if (z[k] < r - i + 1) {
                z[i] = z[k];
            } else {
                l = i;
                while (r < n && str[r - l] == str[r]) {
                    r++;
                }
                z[i] = r - l;
                r--;
            }
        }
    }
}

int z_algorithm_search(const char *text, const char *pattern, int *matches) {
    int text_len = strlen(text);
    int pattern_len = strlen(pattern);
    int concat_len = pattern_len + 1 + text_len;
    
    char *concat = (char*)malloc(concat_len + 1);
    int *z = (int*)malloc(concat_len * sizeof(int));
    
    // Create pattern$text
    sprintf(concat, "%s$%s", pattern, text);
    
    compute_z_array(concat, concat_len, z);
    
    int match_count = 0;
    for (int i = pattern_len + 1; i < concat_len; i++) {
        if (z[i] == pattern_len) {
            matches[match_count++] = i - pattern_len - 1;
        }
    }
    
    free(concat);
    free(z);
    
    return match_count;
}

void generate_text(char *text, int size) {
    unsigned int seed = 42;
    for (int i = 0; i < size - 1; i++) {
        seed = seed * 1103515245 + 12345;
        text[i] = 'a' + (seed % 4);  // Use small alphabet for more matches
    }
    text[size - 1] = '\0';
}

void generate_pattern(char *pattern, int size) {
    unsigned int seed = 123;
    for (int i = 0; i < size - 1; i++) {
        seed = seed * 1103515245 + 12345;
        pattern[i] = 'a' + (seed % 4);
    }
    pattern[size - 1] = '\0';
}

int main() {
    char *text = (char*)malloc(TEXT_SIZE);
    char *pattern = (char*)malloc(PATTERN_SIZE);
    int *matches = (int*)malloc(TEXT_SIZE * sizeof(int));
    
    generate_text(text, TEXT_SIZE);
    generate_pattern(pattern, PATTERN_SIZE);
    
    clock_t start = clock();
    
    int total_matches = 0;
    for (int iter = 0; iter < 20; iter++) {
        total_matches += z_algorithm_search(text, pattern, matches);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Z-algorithm: text=%d, pattern=%d, 20 iterations, %.6f seconds\n",
           TEXT_SIZE, PATTERN_SIZE, time_spent);
    printf("Total matches found: %d\n", total_matches);
    
    free(text);
    free(pattern);
    free(matches);
    
    return 0;
}
