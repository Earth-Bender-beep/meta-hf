// Knuth-Morris-Pratt string matching
// Failure function + linear-time search
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TEXT_SIZE 100000
#define PATTERN_SIZE 50

void compute_lps(const char *pattern, int m, int *lps) {
    int len = 0;
    lps[0] = 0;
    int i = 1;
    
    while (i < m) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
}

int kmp_search(const char *text, int n, const char *pattern, int m, int *matches) {
    int *lps = (int*)malloc(m * sizeof(int));
    compute_lps(pattern, m, lps);
    
    int count = 0;
    int i = 0, j = 0;
    
    while (i < n) {
        if (pattern[j] == text[i]) {
            i++;
            j++;
        }
        
        if (j == m) {
            matches[count++] = i - j;
            j = lps[j - 1];
        } else if (i < n && pattern[j] != text[i]) {
            if (j != 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }
    
    free(lps);
    return count;
}

void generate_text(char *text, int n) {
    const char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    for (int i = 0; i < n; i++) {
        text[i] = alphabet[i % 26];
    }
    text[n] = '\0';
}

void generate_pattern(char *pattern, const char *text, int m) {
    // Create pattern that appears multiple times
    for (int i = 0; i < m; i++) {
        pattern[i] = text[(i * 7) % 26];
    }
    pattern[m] = '\0';
}

int main() {
    char *text = (char*)malloc((TEXT_SIZE + 1) * sizeof(char));
    char *pattern = (char*)malloc((PATTERN_SIZE + 1) * sizeof(char));
    int *matches = (int*)malloc(TEXT_SIZE * sizeof(int));
    
    generate_text(text, TEXT_SIZE);
    generate_pattern(pattern, text, PATTERN_SIZE);
    
    // Insert some actual pattern occurrences
    for (int i = 0; i < 5; i++) {
        int pos = i * (TEXT_SIZE / 6);
        memcpy(text + pos, pattern, PATTERN_SIZE);
    }
    
    clock_t start = clock();
    int count = kmp_search(text, TEXT_SIZE, pattern, PATTERN_SIZE, matches);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("KMP search (text=%d, pattern=%d): %.6f seconds, %d matches\n",
           TEXT_SIZE, PATTERN_SIZE, time_spent, count);
    
    free(text);
    free(pattern);
    free(matches);
    return 0;
}
