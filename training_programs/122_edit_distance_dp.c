// Edit distance (Levenshtein distance) using dynamic programming
// Computes minimum edits (insert, delete, substitute) to transform one string to another
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LEN 500
#define NUM_TESTS 1000

int min3(int a, int b, int c) {
    int min = a;
    if (b < min) min = b;
    if (c < min) min = c;
    return min;
}

int edit_distance(const char *s1, const char *s2) {
    int len1 = strlen(s1);
    int len2 = strlen(s2);
    
    int **dp = (int**)malloc((len1 + 1) * sizeof(int*));
    for (int i = 0; i <= len1; i++) {
        dp[i] = (int*)malloc((len2 + 1) * sizeof(int));
    }
    
    // Initialize base cases
    for (int i = 0; i <= len1; i++) {
        dp[i][0] = i;
    }
    for (int j = 0; j <= len2; j++) {
        dp[0][j] = j;
    }
    
    // Fill DP table
    for (int i = 1; i <= len1; i++) {
        for (int j = 1; j <= len2; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + min3(
                    dp[i-1][j],      // Delete
                    dp[i][j-1],      // Insert
                    dp[i-1][j-1]     // Replace
                );
            }
        }
    }
    
    int result = dp[len1][len2];
    
    for (int i = 0; i <= len1; i++) {
        free(dp[i]);
    }
    free(dp);
    
    return result;
}

void generate_string(char *str, int len) {
    unsigned int seed = 42;
    for (int i = 0; i < len - 1; i++) {
        seed = seed * 1103515245 + 12345;
        str[i] = 'a' + (seed % 26);
    }
    str[len - 1] = '\0';
}

int main() {
    char *str1 = (char*)malloc(MAX_LEN);
    char *str2 = (char*)malloc(MAX_LEN);
    
    clock_t start = clock();
    
    long long total_distance = 0;
    
    for (int test = 0; test < NUM_TESTS; test++) {
        unsigned int seed = test * 123 + 456;
        
        // Generate two strings of varying lengths
        int len1 = 50 + (seed % 100);
        int len2 = 50 + ((seed * 17) % 100);
        
        generate_string(str1, len1);
        generate_string(str2, len2);
        
        int dist = edit_distance(str1, str2);
        total_distance += dist;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Edit distance (DP): %d tests, max_len=%d, %.6f seconds\n",
           NUM_TESTS, MAX_LEN, time_spent);
    printf("Average distance: %.2f\n", total_distance / (double)NUM_TESTS);
    
    free(str1);
    free(str2);
    
    return 0;
}
