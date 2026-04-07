// Manacher's algorithm for finding longest palindromic substring in linear time
// Efficient palindrome detection using center expansion with preprocessing
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TEXT_SIZE 100000

// Preprocess string: insert '#' between characters
char* preprocess(const char *s, int *new_len) {
    int len = strlen(s);
    *new_len = 2 * len + 3;
    char *t = (char*)malloc(*new_len);
    
    t[0] = '^';
    for (int i = 0; i < len; i++) {
        t[2*i + 1] = '#';
        t[2*i + 2] = s[i];
    }
    t[2*len + 1] = '#';
    t[2*len + 2] = '$';
    
    return t;
}

int manacher(const char *s, int *center, int *max_len) {
    int len;
    char *t = preprocess(s, &len);
    int *p = (int*)calloc(len, sizeof(int));
    
    int c = 0, r = 0;  // center and right edge of rightmost palindrome
    
    for (int i = 1; i < len - 1; i++) {
        int mirror = 2 * c - i;
        
        if (i < r) {
            p[i] = (r - i < p[mirror]) ? r - i : p[mirror];
        }
        
        // Try to expand palindrome centered at i
        while (t[i + p[i] + 1] == t[i - p[i] - 1]) {
            p[i]++;
        }
        
        // Update center and right edge
        if (i + p[i] > r) {
            c = i;
            r = i + p[i];
        }
    }
    
    // Find the longest palindrome
    int max_p = 0;
    int center_index = 0;
    for (int i = 1; i < len - 1; i++) {
        if (p[i] > max_p) {
            max_p = p[i];
            center_index = i;
        }
    }
    
    *center = (center_index - 1) / 2 - max_p / 2;
    *max_len = max_p;
    
    free(t);
    free(p);
    
    return max_p;
}

void generate_text_with_palindromes(char *text, int size) {
    unsigned int seed = 42;
    
    // Generate random text
    for (int i = 0; i < size - 1; i++) {
        seed = seed * 1103515245 + 12345;
        text[i] = 'a' + (seed % 8);
    }
    
    // Insert some palindromes
    const char *pals[] = {"racecar", "noon", "level", "deed", "civic"};
    int num_pals = 5;
    
    for (int i = 0; i < 100 && i * 100 < size - 20; i++) {
        seed = seed * 1103515245 + 12345;
        const char *pal = pals[seed % num_pals];
        int pal_len = strlen(pal);
        int pos = (i * 100) % (size - pal_len - 1);
        memcpy(text + pos, pal, pal_len);
    }
    
    text[size - 1] = '\0';
}

int main() {
    char *text = (char*)malloc(TEXT_SIZE);
    generate_text_with_palindromes(text, TEXT_SIZE);
    
    clock_t start = clock();
    
    int total_len = 0;
    for (int iter = 0; iter < 100; iter++) {
        int center, max_len;
        manacher(text, &center, &max_len);
        total_len += max_len;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Manacher's algorithm: text=%d, 100 iterations, %.6f seconds\n",
           TEXT_SIZE, time_spent);
    printf("Average max palindrome length: %d\n", total_len / 100);
    
    free(text);
    return 0;
}
