#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

#define V 2000

int minKey(int key[], int mstSet[]) {
    int min = INT_MAX, min_index;
    
    for (int v = 0; v < V; v++)
        if (mstSet[v] == 0 && key[v] < min)
            min = key[v], min_index = v;
    
    return min_index;
}

void prim(int graph[V][V]) {
    int parent[V];
    int key[V];
    int mstSet[V];
    
    for (int i = 0; i < V; i++)
        key[i] = INT_MAX, mstSet[i] = 0;
    
    key[0] = 0;
    parent[0] = -1;
    
    for (int count = 0; count < V - 1; count++) {
        int u = minKey(key, mstSet);
        mstSet[u] = 1;
        
        for (int v = 0; v < V; v++)
            if (graph[u][v] && mstSet[v] == 0 && graph[u][v] < key[v])
                parent[v] = u, key[v] = graph[u][v];
    }
}

int main() {
    static int graph[V][V];
    
    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            graph[i][j] = 0;
    
    srand(42);
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < 5; j++) {
            int dest = rand() % V;
            if (dest != i) {
                int weight = rand() % 100 + 1;
                graph[i][dest] = weight;
                graph[dest][i] = weight;
            }
        }
    }
    
    clock_t start = clock();
    prim(graph);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Prim MST: %d vertices in %.6f seconds\n", V, time_spent);
    
    return 0;
}
