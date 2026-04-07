// Hungarian algorithm for bipartite matching / assignment problem
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

#define MAX_SIZE 50
#define INF INT_MAX

int min(int a, int b) {
    return (a < b) ? a : b;
}

int hungarian(int cost[][MAX_SIZE], int n) {
    int u[MAX_SIZE], v[MAX_SIZE];
    int p[MAX_SIZE], way[MAX_SIZE];
    
    for (int i = 0; i < n; i++) {
        u[i] = 0;
        v[i] = 0;
        p[i] = 0;
        way[i] = 0;
    }
    
    for (int i = 1; i <= n; i++) {
        p[0] = i;
        int j0 = 0;
        int *minv = (int*)malloc((n + 1) * sizeof(int));
        int *used = (int*)calloc(n + 1, sizeof(int));
        
        for (int j = 0; j <= n; j++) {
            minv[j] = INF;
        }
        
        do {
            used[j0] = 1;
            int i0 = p[j0];
            int delta = INF;
            int j1;
            
            for (int j = 1; j <= n; j++) {
                if (!used[j]) {
                    int cur = cost[i0 - 1][j - 1] - u[i0 - 1] - v[j - 1];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            
            for (int j = 0; j <= n; j++) {
                if (used[j]) {
                    u[p[j] - 1] += delta;
                    v[j - 1] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            
            j0 = j1;
        } while (p[j0] != 0);
        
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
        
        free(minv);
        free(used);
    }
    
    int result = -v[0];
    
    return result;
}

int main() {
    int n = 30;
    int (*cost)[MAX_SIZE] = malloc(sizeof(int[MAX_SIZE][MAX_SIZE]));
    
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            seed = seed * 1103515245 + 12345;
            cost[i][j] = (seed % 100) + 1;
        }
    }
    
    clock_t start = clock();
    int min_cost = hungarian(cost, n);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Hungarian algorithm: %dx%d assignment, %.6f seconds\n", n, n, time_spent);
    printf("Minimum cost: %d\n", min_cost);
    
    free(cost);
    
    return 0;
}
