// Disjoint Set Union (Union-Find) with path compression and union by rank
// Graph connectivity, Kruskal's MST helper
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_ELEMENTS 8000
#define N_OPERATIONS 20000

typedef struct {
    int *parent;
    int *rank;
    int n;
} DisjointSet;

DisjointSet* create_disjoint_set(int n) {
    DisjointSet *ds = (DisjointSet*)malloc(sizeof(DisjointSet));
    ds->n = n;
    ds->parent = (int*)malloc(n * sizeof(int));
    ds->rank = (int*)malloc(n * sizeof(int));
    
    for (int i = 0; i < n; i++) {
        ds->parent[i] = i;
        ds->rank[i] = 0;
    }
    
    return ds;
}

int find(DisjointSet *ds, int x) {
    if (ds->parent[x] != x) {
        ds->parent[x] = find(ds, ds->parent[x]);  // Path compression
    }
    return ds->parent[x];
}

void union_sets(DisjointSet *ds, int x, int y) {
    int root_x = find(ds, x);
    int root_y = find(ds, y);
    
    if (root_x == root_y) return;
    
    // Union by rank
    if (ds->rank[root_x] < ds->rank[root_y]) {
        ds->parent[root_x] = root_y;
    } else if (ds->rank[root_x] > ds->rank[root_y]) {
        ds->parent[root_y] = root_x;
    } else {
        ds->parent[root_y] = root_x;
        ds->rank[root_x]++;
    }
}

int connected(DisjointSet *ds, int x, int y) {
    return find(ds, x) == find(ds, y);
}

int count_components(DisjointSet *ds) {
    int count = 0;
    for (int i = 0; i < ds->n; i++) {
        if (find(ds, i) == i) {
            count++;
        }
    }
    return count;
}

void free_disjoint_set(DisjointSet *ds) {
    free(ds->parent);
    free(ds->rank);
    free(ds);
}

int main() {
    DisjointSet *ds = create_disjoint_set(N_ELEMENTS);
    
    clock_t start = clock();
    
    // Perform union operations
    for (int i = 0; i < N_OPERATIONS / 2; i++) {
        int x = (i * 7) % N_ELEMENTS;
        int y = (i * 13 + 1) % N_ELEMENTS;
        union_sets(ds, x, y);
    }
    
    // Perform find operations
    int connections = 0;
    for (int i = 0; i < N_OPERATIONS / 2; i++) {
        int x = (i * 11) % N_ELEMENTS;
        int y = (i * 17 + 3) % N_ELEMENTS;
        if (connected(ds, x, y)) {
            connections++;
        }
    }
    
    int components = count_components(ds);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Disjoint Set: %d elements, %d operations, %.6f seconds\n",
           N_ELEMENTS, N_OPERATIONS, time_spent);
    printf("Connected pairs: %d, Components: %d\n", connections, components);
    
    free_disjoint_set(ds);
    return 0;
}
