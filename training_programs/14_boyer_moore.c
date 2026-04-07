#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_CHAR 256

void bad_char_heuristic(char* str, int size, int badchar[MAX_CHAR]) {
    for (int i = 0; i < MAX_CHAR; i++)
        badchar[i] = -1;
    for (int i = 0; i < size; i++)
        badchar[(int)str[i]] = i;
}

int boyer_moore(char* text, char* pattern) {
    int m = strlen(pattern);
    int n = strlen(text);
    int badchar[MAX_CHAR];
    int count = 0;
    
    bad_char_heuristic(pattern, m, badchar);
    
    int s = 0;
    while (s <= (n - m)) {
        int j = m - 1;
        
        while (j >= 0 && pattern[j] == text[s + j])
            j--;
        
        if (j < 0) {
            count++;
            s += (s + m < n) ? m - badchar[text[s + m]] : 1;
        } else {
            int shift = j - badchar[text[s + j]];
            s += (shift > 1) ? shift : 1;
        }
    }
    
    return count;
}

int main() {
    int text_size = 500000;
    char* text = (char*)malloc(text_size + 1);
    char* pattern = "ABCABC";
    
    srand(42);
    for (int i = 0; i < text_size; i++) {
        text[i] = 'A' + (rand() % 4);
    }
    text[text_size] = '\0';
    
    clock_t start = clock();
    int matches = boyer_moore(text, pattern);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Boyer-Moore: Found %d matches in %.6f seconds\n", matches, time_spent);
    
    free(text);
    return 0;
}
