// Find Eulerian path/circuit using Hierholzer's algorithm
// Path that visits every edge exactly once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_V 1000
#define MAX_E 5000

typedef struct {
    int adj[MAX_V][MAX_V];
    int degree[MAX_V];
    int in_degree[MAX_V];
    int out_degree[MAX_V];
    int n_vertices;
    int edge_count[MAX_V][MAX_V];
} Graph;

int path[MAX_E];
int path_len;

void graph_init(Graph *g, int n) {
    g->n_vertices = n;
    memset(g->degree, 0, sizeof(g->degree));
    memset(g->in_degree, 0, sizeof(g->in_degree));
    memset(g->out_degree, 0, sizeof(g->out_degree));
    memset(g->edge_count, 0, sizeof(g->edge_count));
}

void add_directed_edge(Graph *g, int u, int v) {
    g->edge_count[u][v]++;
    g->out_degree[u]++;
    g->in_degree[v]++;
    
    // Add to adjacency list if not already present
    int found = 0;
    for (int i = 0; i < g->degree[u]; i++) {
        if (g->adj[u][i] == v) {
            found = 1;
            break;
        }
    }
    if (!found && g->degree[u] < MAX_V) {
        g->adj[u][g->degree[u]++] = v;
    }
}

int can_have_eulerian_path(Graph *g, int *start) {
    int start_vertices = 0, end_vertices = 0;
    *start = 0;
    
    for (int i = 0; i < g->n_vertices; i++) {
        int diff = g->out_degree[i] - g->in_degree[i];
        
        if (diff > 1 || diff < -1) return 0;
        
        if (diff == 1) {
            start_vertices++;
            *start = i;
        } else if (diff == -1) {
            end_vertices++;
        }
    }
    
    return (start_vertices == 0 && end_vertices == 0) || 
           (start_vertices == 1 && end_vertices == 1);
}

void hierholzer_dfs(Graph *g, int u, int stack[], int *stack_top) {
    while (g->degree[u] > 0) {
        int next_idx = g->degree[u] - 1;
        int v = g->adj[u][next_idx];
        
        if (g->edge_count[u][v] > 0) {
            g->edge_count[u][v]--;
            hierholzer_dfs(g, v, stack, stack_top);
        } else {
            g->degree[u]--;
        }
    }
    
    stack[(*stack_top)++] = u;
}

int find_eulerian_path(Graph *g) {
    int start;
    if (!can_have_eulerian_path(g, &start)) {
        return 0;
    }
    
    int stack[MAX_E];
    int stack_top = 0;
    
    hierholzer_dfs(g, start, stack, &stack_top);
    
    path_len = 0;
    for (int i = stack_top - 1; i >= 0; i--) {
        path[path_len++] = stack[i];
    }
    
    return path_len;
}

void generate_eulerian_graph(Graph *g, int n_vertices) {
    graph_init(g, n_vertices);
    
    // Create a graph that definitely has an Eulerian circuit
    // by ensuring all vertices have equal in-degree and out-degree
    for (int i = 0; i < n_vertices; i++) {
        int next = (i + 1) % n_vertices;
        add_directed_edge(g, i, next);
        add_directed_edge(g, i, (i + 2) % n_vertices);
    }
}

int main() {
    Graph g;
    int n_vertices = 800;
    
    generate_eulerian_graph(&g, n_vertices);
    
    int total_edges = 0;
    for (int i = 0; i < n_vertices; i++) {
        total_edges += g->out_degree[i];
    }
    
    clock_t start = clock();
    int path_length = find_eulerian_path(&g);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Eulerian path (Hierholzer): %d vertices, %d edges, %.6f seconds\n",
           n_vertices, total_edges, time_spent);
    printf("Path length: %d\n", path_length);
    
    return 0;
}
