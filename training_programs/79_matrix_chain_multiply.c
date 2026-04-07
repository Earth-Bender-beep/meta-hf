// Matrix chain multiplication using dynamic programming
// Optimal parenthesization problem, 3D DP
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_MATRICES 50
#define INF 999999999

int min(int a, int b) {
    return (a < b) ? a : b;
}

int matrix_chain_order(int *dims, int n) {
    // dp[i][j] = minimum cost to multiply matrices i to j
    int **dp = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        dp[i] = (int*)malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
            dp[i][j] = 0;
        }
    }
    
    // Length is chain length
    for (int len = 2; len < n; len++) {
        for (int i = 1; i < n - len + 1; i++) {
            int j = i + len - 1;
            dp[i][j] = INF;
            
            for (int k = i; k < j; k++) {
                int cost = dp[i][k] + dp[k+1][j] + 
                          dims[i-1] * dims[k] * dims[j];
                
                dp[i][j] = min(dp[i][j], cost);
            }
        }
    }
    
    int result = dp[1][n-1];
    
    for (int i = 0; i < n; i++) {
        free(dp[i]);
    }
    free(dp);
    
    return result;
}

void generate_dimensions(int *dims, int n) {
    for (int i = 0; i < n; i++) {
        dims[i] = 10 + (i % 20);
    }
}

int main() {
    int *dims = (int*)malloc((N_MATRICES + 1) * sizeof(int));
    
    generate_dimensions(dims, N_MATRICES + 1);
    
    clock_t start = clock();
    int min_cost = matrix_chain_order(dims, N_MATRICES + 1);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Matrix chain multiplication: %d matrices, %.6f seconds\n",
           N_MATRICES, time_spent);
    printf("Minimum scalar multiplications: %d\n", min_cost);
    
    free(dims);
    return 0;
}
