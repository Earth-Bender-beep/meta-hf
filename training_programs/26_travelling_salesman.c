#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

#define N 15

int tsp(int graph[N][N], int visited, int pos, int n, int count, int cost, int* ans) {
    if (count == n && graph[pos][0]) {
        if (*ans > cost + graph[pos][0])
            *ans = cost + graph[pos][0];
        return *ans;
    }
    
    for (int i = 0; i < n; i++) {
        if ((visited & (1 << i)) == 0 && graph[pos][i]) {
            tsp(graph, visited | (1 << i), i, n, count + 1, 
                cost + graph[pos][i], ans);
        }
    }
    
    return *ans;
}

int main() {
    int graph[N][N];
    
    srand(42);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j)
                graph[i][j] = 0;
            else
                graph[i][j] = rand() % 50 + 10;
        }
    }
    
    int ans = INT_MAX;
    clock_t start = clock();
    tsp(graph, 1, 0, N, 1, 0, &ans);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("TSP: n=%d, min_cost=%d in %.6f seconds\n", N, ans, time_spent);
    
    return 0;
}
