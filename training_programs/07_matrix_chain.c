#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

int matrix_chain_order(int p[], int n) {
    int** m = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++)
        m[i] = (int*)calloc(n, sizeof(int));
    
    for (int len = 2; len < n; len++) {
        for (int i = 1; i < n - len + 1; i++) {
            int j = i + len - 1;
            m[i][j] = INT_MAX;
            
            for (int k = i; k < j; k++) {
                int cost = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];
                if (cost < m[i][j])
                    m[i][j] = cost;
            }
        }
    }
    
    int result = m[1][n - 1];
    
    for (int i = 0; i < n; i++)
        free(m[i]);
    free(m);
    
    return result;
}

int main() {
    int n = 100;
    int* p = (int*)malloc((n + 1) * sizeof(int));
    
    srand(42);
    for (int i = 0; i <= n; i++) {
        p[i] = rand() % 50 + 10;
    }
    
    clock_t start = clock();
    int result = matrix_chain_order(p, n + 1);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Matrix chain: n=%d, cost=%d in %.6f seconds\n", n, result, time_spent);
    
    free(p);
    return 0;
}
