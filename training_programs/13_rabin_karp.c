#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define d 256
#define q 101

int rabin_karp(char* text, char* pattern) {
    int m = strlen(pattern);
    int n = strlen(text);
    int p = 0;
    int t = 0;
    int h = 1;
    int count = 0;
    
    for (int i = 0; i < m - 1; i++)
        h = (h * d) % q;
    
    for (int i = 0; i < m; i++) {
        p = (d * p + pattern[i]) % q;
        t = (d * t + text[i]) % q;
    }
    
    for (int i = 0; i <= n - m; i++) {
        if (p == t) {
            int j;
            for (j = 0; j < m; j++) {
                if (text[i + j] != pattern[j])
                    break;
            }
            if (j == m)
                count++;
        }
        
        if (i < n - m) {
            t = (d * (t - text[i] * h) + text[i + m]) % q;
            if (t < 0)
                t = (t + q);
        }
    }
    
    return count;
}

int main() {
    int text_size = 500000;
    char* text = (char*)malloc(text_size + 1);
    char* pattern = "ABCDE";
    
    srand(42);
    for (int i = 0; i < text_size; i++) {
        text[i] = 'A' + (rand() % 5);
    }
    text[text_size] = '\0';
    
    clock_t start = clock();
    int matches = rabin_karp(text, pattern);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Rabin-Karp: Found %d matches in %.6f seconds\n", matches, time_spent);
    
    free(text);
    return 0;
}
