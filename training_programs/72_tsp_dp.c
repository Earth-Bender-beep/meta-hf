// Traveling Salesman Problem using dynamic programming (bitmask DP)
// NP-hard problem, exponential complexity, bitmask states
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 18
#define INF 999999

int min(int a, int b) {
    return (a < b) ? a : b;
}

int tsp_dp(int dist[N][N], int n) {
    int max_mask = 1 << n;
    
    // dp[mask][i] = min cost to visit cities in mask, ending at city i
    int **dp = (int**)malloc(max_mask * sizeof(int*));
    for (int i = 0; i < max_mask; i++) {
        dp[i] = (int*)malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
            dp[i][j] = INF;
        }
    }
    
    // Base case: start at city 0
    dp[1][0] = 0;
    
    // Fill DP table
    for (int mask = 1; mask < max_mask; mask++) {
        for (int last = 0; last < n; last++) {
            if (!(mask & (1 << last))) continue;
            if (dp[mask][last] == INF) continue;
            
            for (int next = 0; next < n; next++) {
                if (mask & (1 << next)) continue;
                
                int new_mask = mask | (1 << next);
                dp[new_mask][next] = min(
                    dp[new_mask][next],
                    dp[mask][last] + dist[last][next]
                );
            }
        }
    }
    
    // Find minimum cost to return to start
    int result = INF;
    int final_mask = (1 << n) - 1;
    for (int i = 1; i < n; i++) {
        result = min(result, dp[final_mask][i] + dist[i][0]);
    }
    
    for (int i = 0; i < max_mask; i++) {
        free(dp[i]);
    }
    free(dp);
    
    return result;
}

void create_distance_matrix(int dist[N][N], int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                dist[i][j] = 0;
            } else {
                // Create symmetric distance matrix
                dist[i][j] = 10 + ((i * 7 + j * 13) % 50);
            }
        }
    }
    
    // Make symmetric
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            dist[j][i] = dist[i][j];
        }
    }
}

int main() {
    int (*dist)[N] = malloc(sizeof(int[N][N]));
    
    create_distance_matrix(dist, N);
    
    clock_t start = clock();
    int min_cost = tsp_dp(dist, N);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("TSP (DP): %d cities, %.6f seconds\n", N, time_spent);
    printf("Minimum tour cost: %d\n", min_cost);
    
    free(dist);
    return 0;
}
