// Edit distance (Levenshtein distance) with dynamic programming
// String comparison algorithm, different costs for operations
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LEN 500

int min3(int a, int b, int c) {
    int min = a;
    if (b < min) min = b;
    if (c < min) min = c;
    return min;
}

int edit_distance(const char *s1, int len1, const char *s2, int len2, 
                  int cost_insert, int cost_delete, int cost_replace) {
    int dp[len1 + 1][len2 + 1];
    
    // Initialize base cases
    for (int i = 0; i <= len1; i++) {
        dp[i][0] = i * cost_delete;
    }
    for (int j = 0; j <= len2; j++) {
        dp[0][j] = j * cost_insert;
    }
    
    // Fill DP table
    for (int i = 1; i <= len1; i++) {
        for (int j = 1; j <= len2; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                int replace = dp[i-1][j-1] + cost_replace;
                int delete = dp[i-1][j] + cost_delete;
                int insert = dp[i][j-1] + cost_insert;
                dp[i][j] = min3(replace, delete, insert);
            }
        }
    }
    
    return dp[len1][len2];
}

// Space-optimized version using only 2 rows
int edit_distance_optimized(const char *s1, int len1, const char *s2, int len2) {
    int prev[len2 + 1];
    int curr[len2 + 1];
    
    for (int j = 0; j <= len2; j++) {
        prev[j] = j;
    }
    
    for (int i = 1; i <= len1; i++) {
        curr[0] = i;
        for (int j = 1; j <= len2; j++) {
            if (s1[i-1] == s2[j-1]) {
                curr[j] = prev[j-1];
            } else {
                curr[j] = 1 + min3(prev[j-1], prev[j], curr[j-1]);
            }
        }
        
        // Swap rows
        for (int j = 0; j <= len2; j++) {
            prev[j] = curr[j];
        }
    }
    
    return prev[len2];
}

void generate_similar_strings(char *s1, char *s2, int len, int differences) {
    for (int i = 0; i < len; i++) {
        s1[i] = 'A' + (i % 26);
        s2[i] = s1[i];
    }
    s1[len] = '\0';
    s2[len] = '\0';
    
    // Introduce some differences
    for (int i = 0; i < differences && i < len; i++) {
        int pos = (i * 7) % len;
        s2[pos] = 'A' + ((s2[pos] - 'A' + 13) % 26);
    }
}

int main() {
    char s1[MAX_LEN + 1];
    char s2[MAX_LEN + 1];
    
    generate_similar_strings(s1, s2, MAX_LEN, 50);
    
    clock_t start = clock();
    
    int dist1 = edit_distance(s1, strlen(s1), s2, strlen(s2), 1, 1, 1);
    int dist2 = edit_distance(s1, strlen(s1), s2, strlen(s2), 2, 1, 3);
    int dist3 = edit_distance_optimized(s1, strlen(s1), s2, strlen(s2));
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Edit distance (len=%d): %.6f seconds\n", MAX_LEN, time_spent);
    printf("Standard costs: %d, Custom costs: %d, Optimized: %d\n", dist1, dist2, dist3);
    
    return 0;
}
