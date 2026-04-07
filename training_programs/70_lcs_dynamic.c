// Longest Common Subsequence using dynamic programming
// Classic DP problem, 2D table, traceback
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LEN 800

int max(int a, int b) {
    return (a > b) ? a : b;
}

int lcs_length(const char *s1, int len1, const char *s2, int len2) {
    int **dp = (int**)malloc((len1 + 1) * sizeof(int*));
    for (int i = 0; i <= len1; i++) {
        dp[i] = (int*)malloc((len2 + 1) * sizeof(int));
    }
    
    // Build DP table
    for (int i = 0; i <= len1; i++) {
        for (int j = 0; j <= len2; j++) {
            if (i == 0 || j == 0) {
                dp[i][j] = 0;
            } else if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
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

void lcs_string(const char *s1, int len1, const char *s2, int len2, char *result) {
    int **dp = (int**)malloc((len1 + 1) * sizeof(int*));
    for (int i = 0; i <= len1; i++) {
        dp[i] = (int*)malloc((len2 + 1) * sizeof(int));
    }
    
    // Build DP table
    for (int i = 0; i <= len1; i++) {
        for (int j = 0; j <= len2; j++) {
            if (i == 0 || j == 0) {
                dp[i][j] = 0;
            } else if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    
    // Traceback to find actual LCS
    int i = len1, j = len2;
    int idx = dp[len1][len2];
    result[idx] = '\0';
    
    while (i > 0 && j > 0) {
        if (s1[i-1] == s2[j-1]) {
            result[--idx] = s1[i-1];
            i--;
            j--;
        } else if (dp[i-1][j] > dp[i][j-1]) {
            i--;
        } else {
            j--;
        }
    }
    
    for (int i = 0; i <= len1; i++) {
        free(dp[i]);
    }
    free(dp);
}

void generate_similar_strings(char *s1, char *s2, int len) {
    for (int i = 0; i < len; i++) {
        s1[i] = 'A' + (i % 26);
        // Make s2 similar but with some differences
        if (i % 3 == 0) {
            s2[i] = 'A' + ((i + 5) % 26);
        } else {
            s2[i] = s1[i];
        }
    }
    s1[len] = '\0';
    s2[len] = '\0';
}

int main() {
    char *s1 = (char*)malloc(MAX_LEN + 1);
    char *s2 = (char*)malloc(MAX_LEN + 1);
    char *lcs_result = (char*)malloc(MAX_LEN + 1);
    
    generate_similar_strings(s1, s2, MAX_LEN);
    
    clock_t start = clock();
    int lcs_len = lcs_length(s1, strlen(s1), s2, strlen(s2));
    lcs_string(s1, strlen(s1), s2, strlen(s2), lcs_result);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("LCS: strings of length %d, %.6f seconds\n", MAX_LEN, time_spent);
    printf("LCS length: %d, first 10 chars: %.10s\n", lcs_len, lcs_result);
    
    free(s1);
    free(s2);
    free(lcs_result);
    return 0;
}
