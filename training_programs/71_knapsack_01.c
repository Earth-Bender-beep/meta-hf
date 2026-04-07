// 0/1 Knapsack problem using dynamic programming
// Classic optimization, 2D DP table
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_ITEMS 300
#define CAPACITY 5000

int max(int a, int b) {
    return (a > b) ? a : b;
}

int knapsack_01(int *weights, int *values, int n, int capacity) {
    int **dp = (int**)malloc((n + 1) * sizeof(int*));
    for (int i = 0; i <= n; i++) {
        dp[i] = (int*)calloc(capacity + 1, sizeof(int));
    }
    
    for (int i = 1; i <= n; i++) {
        for (int w = 1; w <= capacity; w++) {
            if (weights[i-1] <= w) {
                dp[i][w] = max(
                    values[i-1] + dp[i-1][w - weights[i-1]],
                    dp[i-1][w]
                );
            } else {
                dp[i][w] = dp[i-1][w];
            }
        }
    }
    
    int result = dp[n][capacity];
    
    for (int i = 0; i <= n; i++) {
        free(dp[i]);
    }
    free(dp);
    
    return result;
}

// Space-optimized version using 1D array
int knapsack_01_optimized(int *weights, int *values, int n, int capacity) {
    int *dp = (int*)calloc(capacity + 1, sizeof(int));
    
    for (int i = 0; i < n; i++) {
        for (int w = capacity; w >= weights[i]; w--) {
            dp[w] = max(dp[w], values[i] + dp[w - weights[i]]);
        }
    }
    
    int result = dp[capacity];
    free(dp);
    return result;
}

void generate_items(int *weights, int *values, int n) {
    for (int i = 0; i < n; i++) {
        weights[i] = 10 + (i % 50);
        values[i] = 5 + (i % 100);
    }
}

int main() {
    int *weights = (int*)malloc(N_ITEMS * sizeof(int));
    int *values = (int*)malloc(N_ITEMS * sizeof(int));
    
    generate_items(weights, values, N_ITEMS);
    
    clock_t start = clock();
    int max_value1 = knapsack_01(weights, values, N_ITEMS, CAPACITY);
    int max_value2 = knapsack_01_optimized(weights, values, N_ITEMS, CAPACITY);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("0/1 Knapsack: %d items, capacity=%d, %.6f seconds\n",
           N_ITEMS, CAPACITY, time_spent);
    printf("Max value (2D): %d, Max value (1D): %d\n", max_value1, max_value2);
    
    free(weights);
    free(values);
    return 0;
}
