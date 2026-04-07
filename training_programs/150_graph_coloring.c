// Graph coloring using greedy and backtracking algorithms
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_NODES 50

typedef struct {
    int adj_matrix[MAX_NODES][MAX_NODES];
    int num_nodes;
} Graph;

void init_graph(Graph *g, int n, unsigned int *seed) {
    g->num_nodes = n;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            g->adj_matrix[i][j] = 0;
        }
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            *seed = *seed * 1103515245 + 12345;
            if ((*seed & 0xFF) < 80) {
                g->adj_matrix[i][j] = 1;
                g->adj_matrix[j][i] = 1;
            }
        }
    }
}

int is_safe(Graph *g, int node, int color, int *colors) {
    for (int i = 0; i < g->num_nodes; i++) {
        if (g->adj_matrix[node][i] && colors[i] == color) {
            return 0;
        }
    }
    return 1;
}

int greedy_coloring(Graph *g, int *colors) {
    for (int i = 0; i < g->num_nodes; i++) {
        colors[i] = -1;
    }
    
    colors[0] = 0;
    
    int max_color = 0;
    
    for (int node = 1; node < g->num_nodes; node++) {
        int *available = (int*)calloc(g->num_nodes, sizeof(int));
        for (int i = 0; i < g->num_nodes; i++) {
            available[i] = 1;
        }
        
        for (int i = 0; i < g->num_nodes; i++) {
            if (g->adj_matrix[node][i] && colors[i] != -1) {
                available[colors[i]] = 0;
            }
        }
        
        int color;
        for (color = 0; color < g->num_nodes; color++) {
            if (available[color]) {
                break;
            }
        }
        
        colors[node] = color;
        if (color > max_color) {
            max_color = color;
        }
        
        free(available);
    }
    
    return max_color + 1;
}

int backtrack_coloring(Graph *g, int *colors, int node, int num_colors) {
    if (node == g->num_nodes) {
        return 1;
    }
    
    for (int color = 0; color < num_colors; color++) {
        if (is_safe(g, node, color, colors)) {
            colors[node] = color;
            
            if (backtrack_coloring(g, colors, node + 1, num_colors)) {
                return 1;
            }
            
            colors[node] = -1;
        }
    }
    
    return 0;
}

int main() {
    Graph g;
    int n = 30;
    
    unsigned int seed = 42;
    init_graph(&g, n, &seed);
    
    int *colors = (int*)malloc(n * sizeof(int));
    
    clock_t start = clock();
    
    int num_colors_greedy = greedy_coloring(&g, colors);
    
    for (int i = 0; i < n; i++) {
        colors[i] = -1;
    }
    
    int num_colors_backtrack = num_colors_greedy;
    while (num_colors_backtrack > 1) {
        if (backtrack_coloring(&g, colors, 0, num_colors_backtrack - 1)) {
            num_colors_backtrack--;
        } else {
            break;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Graph coloring: %d nodes, %.6f seconds\n", n, time_spent);
    printf("Greedy colors: %d, Optimized colors: %d\n",
           num_colors_greedy, num_colors_backtrack);
    
    free(colors);
    
    return 0;
}
