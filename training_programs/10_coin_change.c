#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int coin_change(int coins[], int m, int n) {
    int* table = (int*)calloc(n + 1, sizeof(int));
    table[0] = 1;
    
    for (int i = 0; i < m; i++) {
        for (int j = coins[i]; j <= n; j++) {
            table[j] += table[j - coins[i]];
        }
    }
    
    int result = table[n];
    free(table);
    return result;
}

int main() {
    int coins[] = {1, 2, 5, 10, 20, 50, 100};
    int m = sizeof(coins) / sizeof(coins[0]);
    int n = 5000;
    
    clock_t start = clock();
    int ways = coin_change(coins, m, n);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Coin change: n=%d, ways=%d in %.6f seconds\n", n, ways, time_spent);
    
    return 0;
}
