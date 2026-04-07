// Find bridges (cut edges) in a graph using Tarjan's algorithm
// Critical edges whose removal disconnects the graph
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_V 2000
#define MAX_E 10000

typedef struct Edge {
    int to;
    int id;
} Edge;

typedef struct {
    Edge adj[MAX_V][50];
    int degree[MAX_V];
    int n_vertices;
    int n_edges;
} Graph;

int disc[MAX_V], low[MAX_V], parent[MAX_V];
int visited[MAX_V];
int bridge_edges[MAX_E];
int timer, bridge_count;

void graph_init(Graph *g, int n) {
    g->n_vertices = n;
    g->n_edges = 0;
    memset(g->degree, 0, sizeof(g->degree));
}

void add_edge(Graph *g, int u, int v) {
    if (g->degree[u] < 50 && g->degree[v] < 50) {
        int edge_id = g->n_edges++;
        g->adj[u][g->degree[u]++] = (Edge){v, edge_id};
        g->adj[v][g->degree[v]++] = (Edge){u, edge_id};
    }
}

int min(int a, int b) { return a < b ? a : b; }

void dfs_bridge(Graph *g, int u, int parent_edge_id) {
    visited[u] = 1;
    disc[u] = low[u] = ++timer;
    
    for (int i = 0; i < g->degree[u]; i++) {
        Edge e = g->adj[u][i];
        int v = e.to;
        int edge_id = e.id;
        
        if (!visited[v]) {
            dfs_bridge(g, v, edge_id);
            
            low[u] = min(low[u], low[v]);
            
            // Edge u-v is a bridge if low[v] > disc[u]
            if (low[v] > disc[u]) {
                bridge_edges[edge_id] = 1;
                bridge_count++;
            }
        } else if (edge_id != parent_edge_id) {
            low[u] = min(low[u], disc[v]);
        }
    }
}

int find_bridges(Graph *g) {
    memset(visited, 0, sizeof(visited));
    memset(bridge_edges, 0, sizeof(bridge_edges));
    timer = 0;
    bridge_count = 0;
    
    for (int i = 0; i < g->n_vertices; i++) {
        if (!visited[i]) {
            dfs_bridge(g, i, -1);
        }
    }
    
    return bridge_count;
}

void generate_graph(Graph *g, int n_vertices, int n_edges) {
    graph_init(g, n_vertices);
    
    unsigned int seed = 12345;
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
    int n_edges = 5000;
    
    generate_graph(&g, n_vertices, n_edges);
    
    clock_t start = clock();
    int bridge_cnt = find_bridges(&g);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Bridges (Tarjan): %d vertices, %d edges, %.6f seconds\n",
           n_vertices, g.n_edges, time_spent);
    printf("Found %d bridges\n", bridge_cnt);
    
    return 0;
}
