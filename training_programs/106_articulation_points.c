// Find articulation points (cut vertices) in a graph using Tarjan's algorithm
// Critical vertices whose removal disconnects the graph
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_V 2000
#define MAX_E 8000

typedef struct {
    int adj[MAX_V][MAX_V];
    int degree[MAX_V];
    int n_vertices;
} Graph;

int disc[MAX_V], low[MAX_V], parent[MAX_V];
int visited[MAX_V], is_ap[MAX_V];
int timer;

void graph_init(Graph *g, int n) {
    g->n_vertices = n;
    memset(g->degree, 0, sizeof(g->degree));
}

void add_edge(Graph *g, int u, int v) {
    if (g->degree[u] < MAX_V) {
        g->adj[u][g->degree[u]++] = v;
    }
    if (g->degree[v] < MAX_V) {
        g->adj[v][g->degree[v]++] = u;
    }
}

int min(int a, int b) { return a < b ? a : b; }

void dfs_ap(Graph *g, int u) {
    int children = 0;
    visited[u] = 1;
    disc[u] = low[u] = ++timer;
    
    for (int i = 0; i < g->degree[u]; i++) {
        int v = g->adj[u][i];
        
        if (!visited[v]) {
            children++;
            parent[v] = u;
            dfs_ap(g, v);
            
            low[u] = min(low[u], low[v]);
            
            // u is an articulation point if:
            // 1. u is root and has more than 1 child
            // 2. u is not root and low[v] >= disc[u]
            if (parent[u] == -1 && children > 1) {
                is_ap[u] = 1;
            }
            if (parent[u] != -1 && low[v] >= disc[u]) {
                is_ap[u] = 1;
            }
        } else if (v != parent[u]) {
            low[u] = min(low[u], disc[v]);
        }
    }
}

int find_articulation_points(Graph *g) {
    memset(visited, 0, sizeof(visited));
    memset(is_ap, 0, sizeof(is_ap));
    memset(parent, -1, sizeof(parent));
    timer = 0;
    
    for (int i = 0; i < g->n_vertices; i++) {
        if (!visited[i]) {
            dfs_ap(g, i);
        }
    }
    
    int count = 0;
    for (int i = 0; i < g->n_vertices; i++) {
        if (is_ap[i]) count++;
    }
    
    return count;
}

void generate_graph(Graph *g, int n_vertices, int n_edges) {
    graph_init(g, n_vertices);
    
    unsigned int seed = 42;
    for (int i = 0; i < n_edges; i++) {
        seed = seed * 1103515245 + 12345;
        int u = seed % n_vertices;
        seed = seed * 1103515245 + 12345;
        int v = seed % n_vertices;
        
        if (u != v) {
            add_edge(g, u, v);
        }
    }
}

int main() {
    Graph g;
    int n_vertices = 1500;
    int n_edges = 6000;
    
    generate_graph(&g, n_vertices, n_edges);
    
    clock_t start = clock();
    int ap_count = find_articulation_points(&g);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Articulation points (Tarjan): %d vertices, %d edges, %.6f seconds\n",
           n_vertices, n_edges, time_spent);
    printf("Found %d articulation points\n", ap_count);
    
    return 0;
}
