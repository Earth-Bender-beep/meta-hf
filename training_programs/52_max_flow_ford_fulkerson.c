// Max flow using Ford-Fulkerson with BFS (Edmonds-Karp)
// Graph algorithm, augmenting paths, network flow
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_V 100
#define INF 1000000

int bfs(int capacity[MAX_V][MAX_V], int source, int sink, int parent[], int n) {
    int visited[MAX_V] = {0};
    int queue[MAX_V];
    int front = 0, rear = 0;
    
    queue[rear++] = source;
    visited[source] = 1;
    parent[source] = -1;
    
    while (front < rear) {
        int u = queue[front++];
        
        for (int v = 0; v < n; v++) {
            if (!visited[v] && capacity[u][v] > 0) {
                queue[rear++] = v;
                parent[v] = u;
                visited[v] = 1;
                
                if (v == sink) {
                    return 1;
                }
            }
        }
    }
    
    return 0;
}

int ford_fulkerson(int graph[MAX_V][MAX_V], int source, int sink, int n) {
    int capacity[MAX_V][MAX_V];
    int parent[MAX_V];
    int max_flow = 0;
    
    // Copy graph to capacity matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            capacity[i][j] = graph[i][j];
        }
    }
    
    // Augment flow while path exists
    while (bfs(capacity, source, sink, parent, n)) {
        int path_flow = INF;
        
        // Find minimum capacity in path
        for (int v = sink; v != source; v = parent[v]) {
            int u = parent[v];
            if (capacity[u][v] < path_flow) {
                path_flow = capacity[u][v];
            }
        }
        
        // Update capacities
        for (int v = sink; v != source; v = parent[v]) {
            int u = parent[v];
            capacity[u][v] -= path_flow;
            capacity[v][u] += path_flow;
        }
        
        max_flow += path_flow;
    }
    
    return max_flow;
}

void create_test_graph(int graph[MAX_V][MAX_V], int n) {
    memset(graph, 0, sizeof(int) * MAX_V * MAX_V);
    
    // Create a flow network
    for (int i = 0; i < n-1; i++) {
        // Forward edges
        graph[i][i+1] = 10 + (i % 15);
        
        // Some cross edges
        if (i + 2 < n) {
            graph[i][i+2] = 5 + (i % 8);
        }
        if (i + 3 < n) {
            graph[i][i+3] = 3 + (i % 5);
        }
    }
    
    // Add some backward edges for complexity
    for (int i = 1; i < n; i += 3) {
        if (i > 2) {
            graph[i][i-2] = 7;
        }
    }
}

int main() {
    int graph[MAX_V][MAX_V];
    int n = 50;
    
    create_test_graph(graph, n);
    
    clock_t start = clock();
    int max_flow = ford_fulkerson(graph, 0, n-1, n);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Max flow (Ford-Fulkerson): %d vertices, max_flow=%d, %.6f seconds\n",
           n, max_flow, time_spent);
    
    return 0;
}
