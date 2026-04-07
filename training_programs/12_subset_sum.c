#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int subset_sum(int arr[], int n, int sum) {
    int** dp = (int**)malloc((n + 1) * sizeof(int*));
    for (int i = 0; i <= n; i++)
        dp[i] = (int*)calloc(sum + 1, sizeof(int));
    
    for (int i = 0; i <= n; i++)
        dp[i][0] = 1;
    
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= sum; j++) {
            if (j < arr[i - 1])
                dp[i][j] = dp[i - 1][j];
            else
                dp[i][j] = dp[i - 1][j] || dp[i - 1][j - arr[i - 1]];
        }
    }
    
    int result = dp[n][sum];
    
    for (int i = 0; i <= n; i++)
        free(dp[i]);
    free(dp);
    
    return result;
}

int main() {
    int n = 500;
    int sum = 10000;
    int* arr = (int*)malloc(n * sizeof(int));
    
    srand(42);
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100 + 1;
    }
    
    clock_t start = clock();
    int exists = subset_sum(arr, n, sum);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Subset sum: n=%d, sum=%d, exists=%d in %.6f seconds\n", 
           n, sum, exists, time_spent);
    
    free(arr);
    return 0;
}
