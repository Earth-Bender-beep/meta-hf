#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

typedef struct Edge {
    int src, dest, weight;
} Edge;

typedef struct Graph {
    int V, E;
    Edge* edge;
} Graph;

Graph* createGraph(int V, int E) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = V;
    graph->E = E;
    graph->edge = (Edge*)malloc(E * sizeof(Edge));
    return graph;
}

void bellmanFord(Graph* graph, int src) {
    int V = graph->V;
    int E = graph->E;
    int* dist = (int*)malloc(V * sizeof(int));
    
    for (int i = 0; i < V; i++)
        dist[i] = INT_MAX;
    dist[src] = 0;
    
    for (int i = 1; i <= V - 1; i++) {
        for (int j = 0; j < E; j++) {
            int u = graph->edge[j].src;
            int v = graph->edge[j].dest;
            int weight = graph->edge[j].weight;
            if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
                dist[v] = dist[u] + weight;
        }
    }
    
    free(dist);
}

int main() {
    int V = 3000;
    int E = 15000;
    Graph* graph = createGraph(V, E);
    
    srand(42);
    for (int i = 0; i < E; i++) {
        graph->edge[i].src = rand() % V;
        graph->edge[i].dest = rand() % V;
        graph->edge[i].weight = rand() % 100;
    }
    
    clock_t start = clock();
    bellmanFord(graph, 0);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Bellman-Ford: V=%d, E=%d in %.6f seconds\n", V, E, time_spent);
    
    free(graph->edge);
    free(graph);
    return 0;
}
