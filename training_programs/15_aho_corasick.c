#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXS 5000
#define MAXC 26

int out[MAXS];
int fail[MAXS];
int go_to[MAXS][MAXC];
int states = 1;

void build_goto(char* patterns[], int k) {
    memset(out, 0, sizeof(out));
    memset(go_to, -1, sizeof(go_to));
    
    for (int i = 0; i < k; i++) {
        char* word = patterns[i];
        int current = 0;
        
        for (int j = 0; word[j] != '\0'; j++) {
            int ch = word[j] - 'a';
            
            if (go_to[current][ch] == -1)
                go_to[current][ch] = states++;
            
            current = go_to[current][ch];
        }
        out[current] |= (1 << i);
    }
    
    for (int ch = 0; ch < MAXC; ch++)
        if (go_to[0][ch] == -1)
            go_to[0][ch] = 0;
}

int search_words(char* text, int k) {
    int current = 0;
    int count = 0;
    
    for (int i = 0; text[i] != '\0'; i++) {
        int ch = text[i] - 'a';
        if (ch < 0 || ch >= MAXC)
            continue;
            
        while (current != 0 && go_to[current][ch] == -1)
            current = 0;
        
        if (go_to[current][ch] != -1)
            current = go_to[current][ch];
        
        if (out[current] != 0)
            count++;
    }
    
    return count;
}

int main() {
    char* patterns[] = {"abc", "def", "ghi", "jkl"};
    int k = 4;
    
    int text_size = 100000;
    char* text = (char*)malloc(text_size + 1);
    
    srand(42);
    for (int i = 0; i < text_size; i++) {
        text[i] = 'a' + (rand() % 12);
    }
    text[text_size] = '\0';
    
    build_goto(patterns, k);
    
    clock_t start = clock();
    int matches = search_words(text, k);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Aho-Corasick: Found %d matches in %.6f seconds\n", matches, time_spent);
    
    free(text);
    return 0;
}
