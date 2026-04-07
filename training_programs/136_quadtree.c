// Quadtree for spatial indexing and collision detection
// Hierarchical space partitioning for 2D points
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_POINTS 10000
#define WORLD_SIZE 1000.0
#define QUERY_SIZE 100
#define MAX_CAPACITY 4

typedef struct {
    double x, y;
    int id;
} Point;

typedef struct {
    double x, y, width, height;
} Boundary;

typedef struct QuadTreeNode {
    Boundary boundary;
    Point points[MAX_CAPACITY];
    int count;
    struct QuadTreeNode *nw;
    struct QuadTreeNode *ne;
    struct QuadTreeNode *sw;
    struct QuadTreeNode *se;
    int divided;
} QuadTreeNode;

QuadTreeNode* create_node(Boundary boundary) {
    QuadTreeNode *node = (QuadTreeNode*)malloc(sizeof(QuadTreeNode));
    node->boundary = boundary;
    node->count = 0;
    node->nw = NULL;
    node->ne = NULL;
    node->sw = NULL;
    node->se = NULL;
    node->divided = 0;
    return node;
}

int contains(Boundary *b, Point *p) {
    return (p->x >= b->x - b->width &&
            p->x <= b->x + b->width &&
            p->y >= b->y - b->height &&
            p->y <= b->y + b->height);
}

int intersects(Boundary *b1, Boundary *b2) {
    return !(b2->x - b2->width > b1->x + b1->width ||
             b2->x + b2->width < b1->x - b1->width ||
             b2->y - b2->height > b1->y + b1->height ||
             b2->y + b2->height < b1->y - b1->height);
}

void subdivide(QuadTreeNode *node) {
    double x = node->boundary.x;
    double y = node->boundary.y;
    double w = node->boundary.width / 2;
    double h = node->boundary.height / 2;
    
    Boundary nw_bound = {x - w/2, y + h/2, w, h};
    Boundary ne_bound = {x + w/2, y + h/2, w, h};
    Boundary sw_bound = {x - w/2, y - h/2, w, h};
    Boundary se_bound = {x + w/2, y - h/2, w, h};
    
    node->nw = create_node(nw_bound);
    node->ne = create_node(ne_bound);
    node->sw = create_node(sw_bound);
    node->se = create_node(se_bound);
    node->divided = 1;
}

int insert(QuadTreeNode *node, Point *p) {
    if (!contains(&node->boundary, p)) {
        return 0;
    }
    
    if (node->count < MAX_CAPACITY) {
        node->points[node->count++] = *p;
        return 1;
    }
    
    if (!node->divided) {
        subdivide(node);
    }
    
    if (insert(node->nw, p)) return 1;
    if (insert(node->ne, p)) return 1;
    if (insert(node->sw, p)) return 1;
    if (insert(node->se, p)) return 1;
    
    return 0;
}

void query_range(QuadTreeNode *node, Boundary *range, Point *found, int *count, int max_count) {
    if (!intersects(&node->boundary, range)) {
        return;
    }
    
    for (int i = 0; i < node->count; i++) {
        if (contains(range, &node->points[i])) {
            if (*count < max_count) {
                found[(*count)++] = node->points[i];
            }
        }
    }
    
    if (node->divided) {
        query_range(node->nw, range, found, count, max_count);
        query_range(node->ne, range, found, count, max_count);
        query_range(node->sw, range, found, count, max_count);
        query_range(node->se, range, found, count, max_count);
    }
}

void free_quadtree(QuadTreeNode *node) {
    if (!node) return;
    
    if (node->divided) {
        free_quadtree(node->nw);
        free_quadtree(node->ne);
        free_quadtree(node->sw);
        free_quadtree(node->se);
    }
    
    free(node);
}

void generate_points(Point *points, int n) {
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        points[i].x = ((seed & 0xFFFF) / (double)0xFFFF) * WORLD_SIZE;
        seed = seed * 1103515245 + 12345;
        points[i].y = ((seed & 0xFFFF) / (double)0xFFFF) * WORLD_SIZE;
        points[i].id = i;
    }
}

int main() {
    Boundary world = {WORLD_SIZE/2, WORLD_SIZE/2, WORLD_SIZE/2, WORLD_SIZE/2};
    QuadTreeNode *qtree = create_node(world);
    
    Point *points = (Point*)malloc(NUM_POINTS * sizeof(Point));
    generate_points(points, NUM_POINTS);
    
    clock_t start = clock();
    
    // Insert all points
    for (int i = 0; i < NUM_POINTS; i++) {
        insert(qtree, &points[i]);
    }
    
    // Range queries
    Point *found = (Point*)malloc(1000 * sizeof(Point));
    int total_found = 0;
    
    unsigned int seed = 123;
    for (int q = 0; q < QUERY_SIZE; q++) {
        seed = seed * 1103515245 + 12345;
        double qx = ((seed & 0xFFFF) / (double)0xFFFF) * WORLD_SIZE;
        seed = seed * 1103515245 + 12345;
        double qy = ((seed & 0xFFFF) / (double)0xFFFF) * WORLD_SIZE;
        
        Boundary range = {qx, qy, 50, 50};
        int count = 0;
        query_range(qtree, &range, found, &count, 1000);
        total_found += count;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Quadtree: %d points, %d queries, %.6f seconds\n",
           NUM_POINTS, QUERY_SIZE, time_spent);
    printf("Average points per query: %.2f\n", total_found / (double)QUERY_SIZE);
    
    free_quadtree(qtree);
    free(points);
    free(found);
    
    return 0;
}
