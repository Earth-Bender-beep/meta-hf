#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INF 99999
#define V 400

void floyd_warshall(int graph[V][V], int dist[V][V]) {
    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            dist[i][j] = graph[i][j];
    
    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][k] != INF && dist[k][j] != INF &&
                    dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
}

int main() {
    static int graph[V][V];
    static int dist[V][V];
    
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (i == j)
                graph[i][j] = 0;
            else
                graph[i][j] = INF;
        }
    }
    
    srand(42);
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < 5; j++) {
            int dest = rand() % V;
            if (dest != i)
                graph[i][dest] = rand() % 100 + 1;
        }
    }
    
    clock_t start = clock();
    floyd_warshall(graph, dist);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Floyd-Warshall: %d vertices in %.6f seconds\n", V, time_spent);
    
    return 0;
}
