// Dijkstra's shortest path algorithm with binary heap
// Graph algorithm, priority queue, greedy approach
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define V 500
#define INF 1000000

typedef struct {
    int vertex;
    int dist;
} HeapNode;

typedef struct {
    HeapNode *data;
    int size;
    int *pos;  // Position of vertex in heap
} MinHeap;

MinHeap* create_heap(int capacity) {
    MinHeap *heap = (MinHeap*)malloc(sizeof(MinHeap));
    heap->data = (HeapNode*)malloc(capacity * sizeof(HeapNode));
    heap->pos = (int*)malloc(capacity * sizeof(int));
    heap->size = 0;
    return heap;
}

void swap_nodes(HeapNode *a, HeapNode *b) {
    HeapNode temp = *a;
    *a = *b;
    *b = temp;
}

void heapify(MinHeap *heap, int idx) {
    int smallest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
    
    if (left < heap->size && heap->data[left].dist < heap->data[smallest].dist)
        smallest = left;
    
    if (right < heap->size && heap->data[right].dist < heap->data[smallest].dist)
        smallest = right;
    
    if (smallest != idx) {
        HeapNode *smallest_node = &heap->data[smallest];
        HeapNode *idx_node = &heap->data[idx];
        
        heap->pos[smallest_node->vertex] = idx;
        heap->pos[idx_node->vertex] = smallest;
        
        swap_nodes(smallest_node, idx_node);
        heapify(heap, smallest);
    }
}

HeapNode extract_min(MinHeap *heap) {
    HeapNode root = heap->data[0];
    HeapNode last = heap->data[heap->size - 1];
    heap->data[0] = last;
    
    heap->pos[root.vertex] = heap->size - 1;
    heap->pos[last.vertex] = 0;
    
    heap->size--;
    heapify(heap, 0);
    
    return root;
}

void decrease_key(MinHeap *heap, int vertex, int dist) {
    int i = heap->pos[vertex];
    heap->data[i].dist = dist;
    
    while (i > 0 && heap->data[i].dist < heap->data[(i - 1) / 2].dist) {
        heap->pos[heap->data[i].vertex] = (i - 1) / 2;
        heap->pos[heap->data[(i - 1) / 2].vertex] = i;
        
        swap_nodes(&heap->data[i], &heap->data[(i - 1) / 2]);
        i = (i - 1) / 2;
    }
}

int is_in_heap(MinHeap *heap, int vertex) {
    return heap->pos[vertex] < heap->size;
}

void dijkstra(int graph[V][V], int src, int *dist) {
    MinHeap *heap = create_heap(V);
    
    for (int v = 0; v < V; v++) {
        dist[v] = INF;
        heap->data[v].vertex = v;
        heap->data[v].dist = INF;
        heap->pos[v] = v;
    }
    
    dist[src] = 0;
    decrease_key(heap, src, 0);
    heap->size = V;
    
    while (heap->size > 0) {
        HeapNode min = extract_min(heap);
        int u = min.vertex;
        
        for (int v = 0; v < V; v++) {
            if (graph[u][v] && is_in_heap(heap, v) && 
                dist[u] != INF && dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
                decrease_key(heap, v, dist[v]);
            }
        }
    }
    
    free(heap->data);
    free(heap->pos);
    free(heap);
}

void create_graph(int graph[V][V]) {
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            graph[i][j] = 0;
        }
    }
    
    // Create sparse graph with some edges
    for (int i = 0; i < V - 1; i++) {
        graph[i][i + 1] = 1 + (i % 10);
        graph[i + 1][i] = 1 + (i % 10);
        
        if (i + 5 < V) {
            graph[i][i + 5] = 5 + (i % 7);
            graph[i + 5][i] = 5 + (i % 7);
        }
    }
}

int main() {
    int (*graph)[V] = malloc(sizeof(int[V][V]));
    int *dist = (int*)malloc(V * sizeof(int));
    
    create_graph(graph);
    
    clock_t start = clock();
    dijkstra(graph, 0, dist);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Dijkstra's algorithm: %d vertices, %.6f seconds\n", V, time_spent);
    printf("Shortest distances: dist[1]=%d, dist[%d]=%d\n", dist[1], V-1, dist[V-1]);
    
    free(graph);
    free(dist);
    return 0;
}
