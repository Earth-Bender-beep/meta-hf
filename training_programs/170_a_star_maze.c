// A* pathfinding algorithm for maze solving
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAZE_SIZE 80

typedef struct Node {
    int x, y;
    int g_cost, h_cost, f_cost;
    struct Node *parent;
} Node;

int heuristic(int x1, int y1, int x2, int y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}

Node* find_min_node(Node **open_set, int size) {
    int min_idx = 0;
    for (int i = 1; i < size; i++) {
        if (open_set[i]->f_cost < open_set[min_idx]->f_cost) {
            min_idx = i;
        }
    }
    return open_set[min_idx];
}

int a_star_search(int maze[][MAZE_SIZE], int size, int start_x, int start_y, int goal_x, int goal_y) {
    Node **open_set = (Node**)malloc(size * size * sizeof(Node*));
    int open_size = 0;
    
    int **closed = (int**)calloc(size, sizeof(int*));
    for (int i = 0; i < size; i++) {
        closed[i] = (int*)calloc(size, sizeof(int));
    }
    
    Node *start = (Node*)malloc(sizeof(Node));
    start->x = start_x;
    start->y = start_y;
    start->g_cost = 0;
    start->h_cost = heuristic(start_x, start_y, goal_x, goal_y);
    start->f_cost = start->g_cost + start->h_cost;
    start->parent = NULL;
    
    open_set[open_size++] = start;
    
    int path_length = -1;
    
    while (open_size > 0) {
        Node *current = find_min_node(open_set, open_size);
        
        if (current->x == goal_x && current->y == goal_y) {
            path_length = current->g_cost;
            break;
        }
        
        for (int i = 0; i < open_size; i++) {
            if (open_set[i] == current) {
                open_set[i] = open_set[--open_size];
                break;
            }
        }
        
        closed[current->y][current->x] = 1;
        
        int dirs[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        
        for (int i = 0; i < 4; i++) {
            int nx = current->x + dirs[i][0];
            int ny = current->y + dirs[i][1];
            
            if (nx >= 0 && nx < size && ny >= 0 && ny < size &&
                maze[ny][nx] == 0 && !closed[ny][nx]) {
                
                int g_cost = current->g_cost + 1;
                int h_cost = heuristic(nx, ny, goal_x, goal_y);
                
                Node *neighbor = (Node*)malloc(sizeof(Node));
                neighbor->x = nx;
                neighbor->y = ny;
                neighbor->g_cost = g_cost;
                neighbor->h_cost = h_cost;
                neighbor->f_cost = g_cost + h_cost;
                neighbor->parent = current;
                
                open_set[open_size++] = neighbor;
            }
        }
    }
    
    for (int i = 0; i < size; i++) {
        free(closed[i]);
    }
    free(closed);
    free(open_set);
    
    return path_length;
}

int main() {
    int size = MAZE_SIZE;
    int (*maze)[MAZE_SIZE] = malloc(sizeof(int[MAZE_SIZE][MAZE_SIZE]));
    
    unsigned int seed = 42;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            seed = seed * 1103515245 + 12345;
            maze[y][x] = ((seed & 0xFF) > 200) ? 1 : 0;
        }
    }
    
    maze[0][0] = 0;
    maze[size - 1][size - 1] = 0;
    
    clock_t start = clock();
    int path_length = a_star_search(maze, size, 0, 0, size - 1, size - 1);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("A* pathfinding: %dx%d maze, %.6f seconds\n", size, size, time_spent);
    printf("Path length: %d\n", path_length);
    
    free(maze);
    
    return 0;
}
