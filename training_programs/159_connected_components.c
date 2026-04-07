// Connected components labeling for binary images
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define IMAGE_SIZE 256

typedef struct {
    int parent;
    int rank;
} DisjointSet;

int find(DisjointSet *ds, int x) {
    if (ds[x].parent != x) {
        ds[x].parent = find(ds, ds[x].parent);
    }
    return ds[x].parent;
}

void union_sets(DisjointSet *ds, int x, int y) {
    int root_x = find(ds, x);
    int root_y = find(ds, y);
    
    if (root_x != root_y) {
        if (ds[root_x].rank < ds[root_y].rank) {
            ds[root_x].parent = root_y;
        } else if (ds[root_x].rank > ds[root_y].rank) {
            ds[root_y].parent = root_x;
        } else {
            ds[root_y].parent = root_x;
            ds[root_x].rank++;
        }
    }
}

int connected_components(unsigned char *binary, int *labels, int width, int height) {
    int num_pixels = width * height;
    DisjointSet *ds = (DisjointSet*)malloc(num_pixels * sizeof(DisjointSet));
    
    for (int i = 0; i < num_pixels; i++) {
        ds[i].parent = i;
        ds[i].rank = 0;
        labels[i] = 0;
    }
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            
            if (binary[idx]) {
                if (x > 0 && binary[idx - 1]) {
                    union_sets(ds, idx, idx - 1);
                }
                
                if (y > 0 && binary[idx - width]) {
                    union_sets(ds, idx, idx - width);
                }
            }
        }
    }
    
    int num_components = 0;
    int *component_map = (int*)calloc(num_pixels, sizeof(int));
    
    for (int i = 0; i < num_pixels; i++) {
        if (binary[i]) {
            int root = find(ds, i);
            if (component_map[root] == 0) {
                component_map[root] = ++num_components;
            }
            labels[i] = component_map[root];
        }
    }
    
    free(ds);
    free(component_map);
    
    return num_components;
}

int main() {
    int size = IMAGE_SIZE;
    unsigned char *binary = (unsigned char*)malloc(size * size);
    int *labels = (int*)malloc(size * size * sizeof(int));
    
    unsigned int seed = 42;
    for (int i = 0; i < size * size; i++) {
        seed = seed * 1103515245 + 12345;
        binary[i] = ((seed & 0xFF) > 128) ? 1 : 0;
    }
    
    clock_t start = clock();
    int num_components = connected_components(binary, labels, size, size);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Connected components: %dx%d image, %.6f seconds\n", size, size, time_spent);
    printf("Found %d components\n", num_components);
    
    free(binary);
    free(labels);
    
    return 0;
}
