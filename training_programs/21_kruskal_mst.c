#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct Edge {
    int src, dest, weight;
} Edge;

typedef struct Graph {
    int V, E;
    Edge* edge;
} Graph;

typedef struct Subset {
    int parent;
    int rank;
} Subset;

Graph* createGraph(int V, int E) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = V;
    graph->E = E;
    graph->edge = (Edge*)malloc(E * sizeof(Edge));
    return graph;
}

int find(Subset subsets[], int i) {
    if (subsets[i].parent != i)
        subsets[i].parent = find(subsets, subsets[i].parent);
    return subsets[i].parent;
}

void Union(Subset subsets[], int x, int y) {
    int xroot = find(subsets, x);
    int yroot = find(subsets, y);
    
    if (subsets[xroot].rank < subsets[yroot].rank)
        subsets[xroot].parent = yroot;
    else if (subsets[xroot].rank > subsets[yroot].rank)
        subsets[yroot].parent = xroot;
    else {
        subsets[yroot].parent = xroot;
        subsets[xroot].rank++;
    }
}

int compareEdges(const void* a, const void* b) {
    return ((Edge*)a)->weight - ((Edge*)b)->weight;
}

void kruskal(Graph* graph) {
    int V = graph->V;
    Edge result[V];
    int e = 0;
    int i = 0;
    
    qsort(graph->edge, graph->E, sizeof(Edge), compareEdges);
    
    Subset* subsets = (Subset*)malloc(V * sizeof(Subset));
    for (int v = 0; v < V; v++) {
        subsets[v].parent = v;
        subsets[v].rank = 0;
    }
    
    while (e < V - 1 && i < graph->E) {
        Edge next_edge = graph->edge[i++];
        int x = find(subsets, next_edge.src);
        int y = find(subsets, next_edge.dest);
        
        if (x != y) {
            result[e++] = next_edge;
            Union(subsets, x, y);
        }
    }
    
    free(subsets);
}

int main() {
    int V = 5000;
    int E = 20000;
    Graph* graph = createGraph(V, E);
    
    srand(42);
    for (int i = 0; i < E; i++) {
        graph->edge[i].src = rand() % V;
        graph->edge[i].dest = rand() % V;
        graph->edge[i].weight = rand() % 100;
    }
    
    clock_t start = clock();
    kruskal(graph);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Kruskal MST: V=%d, E=%d in %.6f seconds\n", V, E, time_spent);
    
    free(graph->edge);
    free(graph);
    return 0;
}
