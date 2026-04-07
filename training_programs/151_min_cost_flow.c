// Minimum cost flow - cost-constrained network flow
#include <stdio.h>
<stdlib.h>
#include <limits.h>
#include <time.h>

#define MAX_NODES 100
#define MAX_EDGES 500
#define INF INT_MAX

typedef struct {
    int from, to;
    int capacity;
    int cost;
    int flow;
} Edge;

typedef struct {
    Edge edges[MAX_EDGES];
    int num_edges;
    int num_nodes;
} FlowNetwork;

void init_network(FlowNetwork *net, int nodes) {
    net->num_nodes = nodes;
    net->num_edges = 0;
}

void add_edge(FlowNetwork *net, int from, int to, int capacity, int cost) {
    if (net->num_edges < MAX_EDGES) {
        net->edges[net->num_edges].from = from;
        net->edges[net->num_edges].to = to;
        net->edges[net->num_edges].capacity = capacity;
        net->edges[net->num_edges].cost = cost;
        net->edges[net->num_edges].flow = 0;
        net->num_edges++;
    }
}

int bellman_ford(FlowNetwork *net, int source, int sink, int *dist, int *parent) {
    for (int i = 0; i < net->num_nodes; i++) {
        dist[i] = INF;
        parent[i] = -1;
    }
    dist[source] = 0;
    
    for (int iter = 0; iter < net->num_nodes - 1; iter++) {
        for (int e = 0; e < net->num_edges; e++) {
            int u = net->edges[e].from;
            int v = net->edges[e].to;
            int residual = net->edges[e].capacity - net->edges[e].flow;
            
            if (residual > 0 && dist[u] != INF) {
                if (dist[u] + net->edges[e].cost < dist[v]) {
                    dist[v] = dist[u] + net->edges[e].cost;
                    parent[v] = e;
                }
            }
        }
    }
    
    return dist[sink] != INF;
}

int min_cost_flow(FlowNetwork *net, int source, int sink, int max_flow) {
    int total_cost = 0;
    int *dist = (int*)malloc(net->num_nodes * sizeof(int));
    int *parent = (int*)malloc(net->num_nodes * sizeof(int));
    
    while (max_flow > 0 && bellman_ford(net, source, sink, dist, parent)) {
        int flow = max_flow;
        
        for (int v = sink; v != source; v = net->edges[parent[v]].from) {
            int e = parent[v];
            int residual = net->edges[e].capacity - net->edges[e].flow;
            if (residual < flow) {
                flow = residual;
            }
        }
        
        for (int v = sink; v != source; v = net->edges[parent[v]].from) {
            int e = parent[v];
            net->edges[e].flow += flow;
            total_cost += flow * net->edges[e].cost;
        }
        
        max_flow -= flow;
    }
    
    free(dist);
    free(parent);
    
    return total_cost;
}

int main() {
    FlowNetwork net;
    init_network(&net, 20);
    
    unsigned int seed = 42;
    for (int i = 0; i < 15; i++) {
        seed = seed * 1103515245 + 12345;
        int from = seed % 18;
        seed = seed * 1103515245 + 12345;
        int to = (from + 1 + seed % 5) % 20;
        seed = seed * 1103515245 + 12345;
        int capacity = (seed % 20) + 10;
        seed = seed * 1103515245 + 12345;
        int cost = (seed % 10) + 1;
        
        add_edge(&net, from, to, capacity, cost);
    }
    
    clock_t start = clock();
    
    int total_cost = min_cost_flow(&net, 0, 19, 50);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Min cost flow: %d nodes, %d edges, %.6f seconds\n",
           net.num_nodes, net.num_edges, time_spent);
    printf("Total cost: %d\n", total_cost);
    
    return 0;
}
