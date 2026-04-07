// Maze generation using recursive backtracking
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAZE_SIZE 50

typedef struct {
    int visited;
    int walls[4];  // top, right, bottom, left
} Cell;

void init_maze(Cell maze[][MAZE_SIZE], int size) {
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            maze[y][x].visited = 0;
            for (int i = 0; i < 4; i++) {
                maze[y][x].walls[i] = 1;
            }
        }
    }
}

void recursive_backtrack(Cell maze[][MAZE_SIZE], int x, int y, int size, unsigned int *seed) {
    maze[y][x].visited = 1;
    
    int dirs[4][2] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
    
    for (int i = 0; i < 4; i++) {
        *seed = *seed * 1103515245 + 12345;
        int j = (*seed) % 4;
        int temp_x = dirs[i][0];
        int temp_y = dirs[i][1];
        dirs[i][0] = dirs[j][0];
        dirs[i][1] = dirs[j][1];
        dirs[j][0] = temp_x;
        dirs[j][1] = temp_y;
    }
    
    for (int i = 0; i < 4; i++) {
        int nx = x + dirs[i][0];
        int ny = y + dirs[i][1];
        
        if (nx >= 0 && nx < size && ny >= 0 && ny < size && !maze[ny][nx].visited) {
            if (dirs[i][0] == 1) {
                maze[y][x].walls[1] = 0;
                maze[ny][nx].walls[3] = 0;
            } else if (dirs[i][0] == -1) {
                maze[y][x].walls[3] = 0;
                maze[ny][nx].walls[1] = 0;
            } else if (dirs[i][1] == 1) {
                maze[y][x].walls[2] = 0;
                maze[ny][nx].walls[0] = 0;
            } else if (dirs[i][1] == -1) {
                maze[y][x].walls[0] = 0;
                maze[ny][nx].walls[2] = 0;
            }
            
            recursive_backtrack(maze, nx, ny, size, seed);
        }
    }
}

int main() {
    int size = MAZE_SIZE;
    Cell (*maze)[MAZE_SIZE] = malloc(sizeof(Cell[MAZE_SIZE][MAZE_SIZE]));
    
    init_maze(maze, size);
    
    unsigned int seed = 42;
    
    clock_t start = clock();
    recursive_backtrack(maze, 0, 0, size, &seed);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    int wall_count = 0;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            for (int i = 0; i < 4; i++) {
                wall_count += maze[y][x].walls[i];
            }
        }
    }
    
    printf("Maze generation: %dx%d, %.6f seconds\n", size, size, time_spent);
    printf("Total walls: %d\n", wall_count);
    
    free(maze);
    
    return 0;
}
