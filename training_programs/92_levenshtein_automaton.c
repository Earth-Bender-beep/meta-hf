// Levenshtein automaton for fuzzy string matching
// Dynamic programming state machine, edit distance queries
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_WORD_LEN 50
#define NUM_WORDS 2000
#define MAX_DISTANCE 2

int min3(int a, int b, int c) {
    int min = a;
    if (b < min) min = b;
    if (c < min) min = c;
    return min;
}

int levenshtein_distance(const char *s1, const char *s2) {
    int len1 = strlen(s1);
    int len2 = strlen(s2);
    
    int *prev = (int*)malloc((len2 + 1) * sizeof(int));
    int *curr = (int*)malloc((len2 + 1) * sizeof(int));
    
    for (int j = 0; j <= len2; j++) {
        prev[j] = j;
    }
    
    for (int i = 1; i <= len1; i++) {
        curr[0] = i;
        
        for (int j = 1; j <= len2; j++) {
            int cost = (s1[i-1] == s2[j-1]) ? 0 : 1;
            curr[j] = min3(
                prev[j] + 1,      // deletion
                curr[j-1] + 1,    // insertion
                prev[j-1] + cost  // substitution
            );
        }
        
        int *temp = prev;
        prev = curr;
        curr = temp;
    }
    
    int result = prev[len2];
    free(prev);
    free(curr);
    
    return result;
}

int fuzzy_search(const char *pattern, char **dictionary, int dict_size, 
                 int max_distance, int *matches) {
    int match_count = 0;
    
    for (int i = 0; i < dict_size; i++) {
        int distance = levenshtein_distance(pattern, dictionary[i]);
        if (distance <= max_distance) {
            matches[match_count++] = i;
        }
    }
    
    return match_count;
}

void generate_word(char *word, int len, int seed) {
    for (int i = 0; i < len - 1; i++) {
        word[i] = 'a' + ((seed * 13 + i * 7) % 26);
    }
    word[len - 1] = '\0';
}

int main() {
    char **dictionary = (char**)malloc(NUM_WORDS * sizeof(char*));
    int *matches = (int*)malloc(NUM_WORDS * sizeof(int));
    
    for (int i = 0; i < NUM_WORDS; i++) {
        dictionary[i] = (char*)malloc(MAX_WORD_LEN);
        generate_word(dictionary[i], 10, i);
    }
    
    char pattern[MAX_WORD_LEN];
    generate_word(pattern, 10, 42);
    
    clock_t start = clock();
    
    int total_matches = 0;
    for (int iter = 0; iter < 50; iter++) {
        int match_count = fuzzy_search(pattern, dictionary, NUM_WORDS, 
                                      MAX_DISTANCE, matches);
        total_matches += match_count;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Levenshtein automaton: %d words, %.6f seconds\n", 
           NUM_WORDS, time_spent);
    printf("Total fuzzy matches (50 iterations): %d\n", total_matches);
    
    for (int i = 0; i < NUM_WORDS; i++) {
        free(dictionary[i]);
    }
    free(dictionary);
    free(matches);
    
    return 0;
}
