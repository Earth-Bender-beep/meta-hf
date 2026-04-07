// Hopcroft–Karp algorithm for maximum bipartite matching
// Graph algorithm with BFS/DFS layers; efficient O(E * sqrt(V))
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NL 600
#define NR 600
#define MAXE 8000
#define INF 1000000000

typedef struct Edge { int v; int next; } Edge;

typedef struct {
    int head[NL+1];
    Edge edges[MAXE];
    int edge_cnt;
} Graph;

void graph_init(Graph *g) {
    for (int i = 1; i <= NL; i++) g->head[i] = -1;
    g->edge_cnt = 0;
}

void add_edge(Graph *g, int u, int v) {
    if (g->edge_cnt >= MAXE) return;
    g->edges[g->edge_cnt] = (Edge){v, g->head[u]};
    g->head[u] = g->edge_cnt++;
}

int pairU[NL+1], pairV[NR+1], distArr[NL+1];

int bfs(Graph *g) {
    static int queue[NL+1];
    int qh = 0, qt = 0;
    for (int u = 1; u <= NL; u++) {
        if (pairU[u] == 0) { distArr[u] = 0; queue[qt++] = u; }
        else distArr[u] = INF;
    }
    int found = 0;
    while (qh < qt) {
        int u = queue[qh++];
        for (int ei = g->head[u]; ei != -1; ei = g->edges[ei].next) {
            int v = g->edges[ei].v;
            int pu = pairV[v];
            if (pu == 0) found = 1; // free vertex on right reachable
            else if (distArr[pu] == INF) {
                distArr[pu] = distArr[u] + 1;
                queue[qt++] = pu;
            }
        }
    }
    return found;
}

int dfs(Graph *g, int u) {
    for (int ei = g->head[u]; ei != -1; ei = g->edges[ei].next) {
        int v = g->edges[ei].v;
        int pu = pairV[v];
        if (pu == 0 || (distArr[pu] == distArr[u] + 1 && dfs(g, pu))) {
            pairU[u] = v;
            pairV[v] = u;
            return 1;
        }
    }
    distArr[u] = INF;
    return 0;
}

int hopcroft_karp(Graph *g) {
    for (int i = 1; i <= NL; i++) pairU[i] = 0;
    for (int i = 1; i <= NR; i++) pairV[i] = 0;
    int matching = 0;
    while (bfs(g)) {
        for (int u = 1; u <= NL; u++) {
            if (pairU[u] == 0) matching += dfs(g, u);
        }
    }
    return matching;
}

void generate_bipartite_graph(Graph *g) {
    // Deterministic sparse bipartite graph
    for (int u = 1; u <= NL; u++) {
        int deg = 1 + (u % 7);
        for (int k = 0; k < deg; k++) {
            int v = 1 + ((u * 37 + k * 13) % NR);
            add_edge(g, u, v);
        }
    }
}

int main() {
    Graph g; graph_init(&g);
    generate_bipartite_graph(&g);

    clock_t start = clock();
    int maxmatch = hopcroft_karp(&g);
    clock_t end = clock();

    double secs = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Hopcroft–Karp: NL=%d NR=%d, edges=%d, match=%d, %.6f sec\n",
           NL, NR, g.edge_cnt, maxmatch, secs);
    return 0;
}
