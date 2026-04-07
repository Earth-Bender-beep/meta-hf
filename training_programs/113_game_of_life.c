// Conway's Game of Life - cellular automaton simulation
// Grid-based simulation with birth/survival rules
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GRID_SIZE 200
#define GENERATIONS 500

typedef struct {
    int cells[GRID_SIZE][GRID_SIZE];
    int width;
    int height;
} Grid;

void init_grid(Grid *g, int width, int height) {
    g->width = width;
    g->height = height;
    memset(g->cells, 0, sizeof(g->cells));
}

void set_cell(Grid *g, int x, int y, int value) {
    if (x >= 0 && x < g->width && y >= 0 && y < g->height) {
        g->cells[y][x] = value;
    }
}

int get_cell(Grid *g, int x, int y) {
    if (x < 0 || x >= g->width || y < 0 || y >= g->height) {
        return 0;
    }
    return g->cells[y][x];
}

int count_neighbors(Grid *g, int x, int y) {
    int count = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            count += get_cell(g, x + dx, y + dy);
        }
    }
    return count;
}

void step(Grid *current, Grid *next) {
    for (int y = 0; y < current->height; y++) {
        for (int x = 0; x < current->width; x++) {
            int neighbors = count_neighbors(current, x, y);
            int cell = get_cell(current, x, y);
            
            // Conway's rules:
            // 1. Any live cell with 2-3 neighbors survives
            // 2. Any dead cell with exactly 3 neighbors becomes alive
            // 3. All other cells die or stay dead
            
            if (cell) {
                next->cells[y][x] = (neighbors == 2 || neighbors == 3) ? 1 : 0;
            } else {
                next->cells[y][x] = (neighbors == 3) ? 1 : 0;
            }
        }
    }
}

int count_alive(Grid *g) {
    int count = 0;
    for (int y = 0; y < g->height; y++) {
        for (int x = 0; x < g->width; x++) {
            count += g->cells[y][x];
        }
    }
    return count;
}

void init_random_pattern(Grid *g) {
    unsigned int seed = 42;
    for (int y = 0; y < g->height; y++) {
        for (int x = 0; x < g->width; x++) {
            seed = seed * 1103515245 + 12345;
            g->cells[y][x] = (seed % 100) < 30 ? 1 : 0;  // 30% alive
        }
    }
}

int main() {
    Grid grid1, grid2;
    init_grid(&grid1, GRID_SIZE, GRID_SIZE);
    init_grid(&grid2, GRID_SIZE, GRID_SIZE);
    
    init_random_pattern(&grid1);
    
    int initial_alive = count_alive(&grid1);
    
    clock_t start = clock();
    
    Grid *current = &grid1;
    Grid *next = &grid2;
    
    for (int gen = 0; gen < GENERATIONS; gen++) {
        step(current, next);
        
        // Swap grids
        Grid *temp = current;
        current = next;
        next = temp;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    int final_alive = count_alive(current);
    
    printf("Game of Life: %dx%d grid, %d generations, %.6f seconds\n",
           GRID_SIZE, GRID_SIZE, GENERATIONS, time_spent);
    printf("Initial alive: %d, Final alive: %d\n", initial_alive, final_alive);
    
    return 0;
}
