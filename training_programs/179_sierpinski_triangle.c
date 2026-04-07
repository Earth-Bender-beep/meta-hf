// Sierpinski triangle fractal generation using chaos game
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ITERATIONS 100000
#define GRID_SIZE 512

typedef struct {
    double x, y;
} Point;

void generate_sierpinski_chaos(unsigned char *grid, int size, int iterations) {
    Point vertices[3] = {
        {size / 2.0, 0},
        {0, size - 1},
        {size - 1, size - 1}
    };
    
    Point current = {size / 2.0, size / 2.0};
    unsigned int seed = 42;
    
    for (int i = 0; i < iterations; i++) {
        seed = seed * 1103515245 + 12345;
        int vertex_idx = seed % 3;
        
        current.x = (current.x + vertices[vertex_idx].x) / 2.0;
        current.y = (current.y + vertices[vertex_idx].y) / 2.0;
        
        int x = (int)current.x;
        int y = (int)current.y;
        
        if (x >= 0 && x < size && y >= 0 && y < size) {
            grid[y * size + x] = 255;
        }
    }
}

int main() {
    int size = GRID_SIZE;
    unsigned char *grid = (unsigned char*)calloc(size * size, sizeof(unsigned char));
    
    clock_t start = clock();
    generate_sierpinski_chaos(grid, size, ITERATIONS);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    int filled_pixels = 0;
    for (int i = 0; i < size * size; i++) {
        if (grid[i] > 0) filled_pixels++;
    }
    
    printf("Sierpinski triangle: %dx%d, %d iterations, %.6f seconds\n",
           size, size, ITERATIONS, time_spent);
    printf("Filled pixels: %d\n", filled_pixels);
    
    free(grid);
    return 0;
}
