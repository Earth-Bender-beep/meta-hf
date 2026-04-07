// Stoer-Wagner algorithm for minimum cut in undirected graphs
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

#define MAX_NODES 100

typedef struct {
    int **weights;
    int num_nodes;
    int *merged;
} Graph;

Graph* create_graph(int n) {
    Graph *g = (Graph*)malloc(sizeof(Graph));
    g->num_nodes = n;
    g->weights = (int**)malloc(n * sizeof(int*));
    g->merged = (int*)calloc(n, sizeof(int));
    
    for (int i = 0; i < n; i++) {
        g->weights[i] = (int*)calloc(n, sizeof(int));
    }
    
    return g;
}

void add_edge(Graph *g, int u, int v, int weight) {
    g->weights[u][v] += weight;
    g->weights[v][u] += weight;
}

int find_max_adjacent(Graph *g, int *in_a, int *key) {
    int max_val = -1;
    int max_idx = -1;
    
    for (int i = 0; i < g->num_nodes; i++) {
        if (!g->merged[i] && !in_a[i] && key[i] > max_val) {
            max_val = key[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

int minimum_cut_phase(Graph *g, int *s, int *t) {
    int *in_a = (int*)calloc(g->num_nodes, sizeof(int));
    int *key = (int*)calloc(g->num_nodes, sizeof(int));
    
    *s = -1;
    *t = -1;
    
    for (int phase = 0; phase < g->num_nodes; phase++) {
        if (g->merged[phase]) continue;
        
        int v = find_max_adjacent(g, in_a, key);
        if (v == -1) break;
        
        in_a[v] = 1;
        *s = *t;
        *t = v;
        
        for (int i = 0; i < g->num_nodes; i++) {
            if (!g->merged[i] && !in_a[i]) {
                key[i] += g->weights[v][i];
            }
        }
    }
    
    int cut_weight = 0;
    for (int i = 0; i < g->num_nodes; i++) {
        if (!g->merged[i]) {
            cut_weight += g->weights[*t][i];
        }
    }
    
    free(in_a);
    free(key);
    
    return cut_weight;
}

void merge_nodes(Graph *g, int s, int t) {
    for (int i = 0; i < g->num_nodes; i++) {
        g->weights[s][i] += g->weights[t][i];
        g->weights[i][s] += g->weights[i][t];
    }
    
    g->merged[t] = 1;
}

int stoer_wagner(Graph *g) {
    int min_cut = INT_MAX;
    
    for (int phase = 0; phase < g->num_nodes - 1; phase++) {
        int s, t;
        int cut_weight = minimum_cut_phase(g, &s, &t);
        
        if (t != -1 && cut_weight < min_cut) {
            min_cut = cut_weight;
        }
        
        if (s != -1 && t != -1) {
            merge_nodes(g, s, t);
        }
    }
    
    return min_cut;
}

int main() {
    int n = 30;
    Graph *g = create_graph(n);
    
    unsigned int seed = 42;
    
    // Create a connected graph
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            seed = seed * 1103515245 + 12345;
            if ((seed & 0xFF) < 60) {
                int weight = (seed % 10) + 1;
                add_edge(g, i, j, weight);
            }
        }
    }
    
    // Ensure connectivity
    for (int i = 0; i < n - 1; i++) {
        if (g->weights[i][i + 1] == 0) {
            add_edge(g, i, i + 1, 1);
        }
    }
    
    clock_t start = clock();
    int min_cut = stoer_wagner(g);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Stoer-Wagner Min-Cut: %d nodes, min cut = %d, %.6f seconds\n",
           n, min_cut, time_spent);
    
    for (int i = 0; i < n; i++) {
        free(g->weights[i]);
    }
    free(g->weights);
    free(g->merged);
    free(g);
    
    return 0;
}
