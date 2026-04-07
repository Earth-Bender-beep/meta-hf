// A* pathfinding algorithm on a grid
// Heuristic search, priority queue, grid navigation
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define GRID_SIZE 100
#define OBSTACLE_DENSITY 0.2

typedef struct {
    int x, y;
    int g_cost;  // Cost from start
    int h_cost;  // Heuristic cost to end
    int f_cost;  // Total cost
    struct Node *parent;
} Node;

typedef struct {
    Node **data;
    int size;
    int capacity;
} PriorityQueue;

PriorityQueue* create_pq(int capacity) {
    PriorityQueue *pq = (PriorityQueue*)malloc(sizeof(PriorityQueue));
    pq->data = (Node**)malloc(capacity * sizeof(Node*));
    pq->size = 0;
    pq->capacity = capacity;
    return pq;
}

void pq_push(PriorityQueue *pq, Node *node) {
    if (pq->size >= pq->capacity) return;
    
    int i = pq->size++;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (pq->data[parent]->f_cost <= node->f_cost) break;
        pq->data[i] = pq->data[parent];
        i = parent;
    }
    pq->data[i] = node;
}

Node* pq_pop(PriorityQueue *pq) {
    if (pq->size == 0) return NULL;
    
    Node *result = pq->data[0];
    Node *last = pq->data[--pq->size];
    
    int i = 0;
    while (2 * i + 1 < pq->size) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int smallest = left;
        
        if (right < pq->size && pq->data[right]->f_cost < pq->data[left]->f_cost) {
            smallest = right;
        }
        
        if (last->f_cost <= pq->data[smallest]->f_cost) break;
        
        pq->data[i] = pq->data[smallest];
        i = smallest;
    }
    pq->data[i] = last;
    
    return result;
}

int manhattan_distance(int x1, int y1, int x2, int y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}

int a_star(int **grid, int size, int start_x, int start_y, int end_x, int end_y) {
    Node ***nodes = (Node***)malloc(size * sizeof(Node**));
    for (int i = 0; i < size; i++) {
        nodes[i] = (Node**)calloc(size, sizeof(Node*));
    }
    
    PriorityQueue *open_set = create_pq(size * size);
    int **closed = (int**)calloc(size, sizeof(int*));
    for (int i = 0; i < size; i++) {
        closed[i] = (int*)calloc(size, sizeof(int));
    }
    
    Node *start = (Node*)malloc(sizeof(Node));
    start->x = start_x;
    start->y = start_y;
    start->g_cost = 0;
    start->h_cost = manhattan_distance(start_x, start_y, end_x, end_y);
    start->f_cost = start->h_cost;
    start->parent = NULL;
    
    pq_push(open_set, start);
    nodes[start_x][start_y] = start;
    
    int path_length = -1;
    int dx[] = {0, 1, 0, -1};
    int dy[] = {1, 0, -1, 0};
    
    while (open_set->size > 0) {
        Node *current = pq_pop(open_set);
        
        if (current->x == end_x && current->y == end_y) {
            path_length = current->g_cost;
            break;
        }
        
        closed[current->x][current->y] = 1;
        
        for (int i = 0; i < 4; i++) {
            int nx = current->x + dx[i];
            int ny = current->y + dy[i];
            
            if (nx < 0 || nx >= size || ny < 0 || ny >= size) continue;
            if (grid[nx][ny] == 1 || closed[nx][ny]) continue;
            
            int new_g = current->g_cost + 1;
            
            if (!nodes[nx][ny] || new_g < nodes[nx][ny]->g_cost) {
                if (!nodes[nx][ny]) {
                    nodes[nx][ny] = (Node*)malloc(sizeof(Node));
                    nodes[nx][ny]->x = nx;
                    nodes[nx][ny]->y = ny;
                }
                
                nodes[nx][ny]->g_cost = new_g;
                nodes[nx][ny]->h_cost = manhattan_distance(nx, ny, end_x, end_y);
                nodes[nx][ny]->f_cost = new_g + nodes[nx][ny]->h_cost;
                nodes[nx][ny]->parent = current;
                
                pq_push(open_set, nodes[nx][ny]);
            }
        }
    }
    
    // Cleanup
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (nodes[i][j]) free(nodes[i][j]);
        }
        free(nodes[i]);
        free(closed[i]);
    }
    free(nodes);
    free(closed);
    free(open_set->data);
    free(open_set);
    
    return path_length;
}

int main() {
    int **grid = (int**)malloc(GRID_SIZE * sizeof(int*));
    for (int i = 0; i < GRID_SIZE; i++) {
        grid[i] = (int*)calloc(GRID_SIZE, sizeof(int));
    }
    
    // Create obstacles (deterministic)
    unsigned int seed = 12345;
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            seed = seed * 1103515245 + 12345;
            double rand_val = (seed & 0x7FFFFFFF) / (double)0x7FFFFFFF;
            if (rand_val < OBSTACLE_DENSITY) {
                grid[i][j] = 1;
            }
        }
    }
    
    grid[0][0] = 0;
    grid[GRID_SIZE-1][GRID_SIZE-1] = 0;
    
    clock_t start = clock();
    int path_length = a_star(grid, GRID_SIZE, 0, 0, GRID_SIZE-1, GRID_SIZE-1);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("A* pathfinding: %dx%d grid, %.6f seconds\n", 
           GRID_SIZE, GRID_SIZE, time_spent);
    printf("Path length: %d\n", path_length);
    
    for (int i = 0; i < GRID_SIZE; i++) {
        free(grid[i]);
    }
    free(grid);
    
    return 0;
}
