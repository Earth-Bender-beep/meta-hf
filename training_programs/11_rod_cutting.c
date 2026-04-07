#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int max(int a, int b) {
    return (a > b) ? a : b;
}

int rod_cutting(int prices[], int n) {
    int* val = (int*)calloc(n + 1, sizeof(int));
    
    for (int i = 1; i <= n; i++) {
        int max_val = -1;
        for (int j = 0; j < i; j++) {
            max_val = max(max_val, prices[j] + val[i - j - 1]);
        }
        val[i] = max_val;
    }
    
    int result = val[n];
    free(val);
    return result;
}

int main() {
    int n = 10000;
    int* prices = (int*)malloc(n * sizeof(int));
    
    srand(42);
    for (int i = 0; i < n; i++) {
        prices[i] = rand() % 100 + 1;
    }
    
    clock_t start = clock();
    int max_profit = rod_cutting(prices, n);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Rod cutting: n=%d, profit=%d in %.6f seconds\n", n, max_profit, time_spent);
    
    free(prices);
    return 0;
}
