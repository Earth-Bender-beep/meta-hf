// Quadtree for spatial partitioning and range queries
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_POINTS_PER_NODE 4
#define NUM_POINTS 5000

typedef struct Point {
    double x, y;
    struct Point *next;
} Point;

typedef struct QuadTree {
    double x, y, width, height;
    Point *points;
    int point_count;
    struct QuadTree *nw, *ne, *sw, *se;
} QuadTree;

QuadTree* create_quadtree(double x, double y, double width, double height) {
    QuadTree *node = (QuadTree*)malloc(sizeof(QuadTree));
    node->x = x;
    node->y = y;
    node->width = width;
    node->height = height;
    node->points = NULL;
    node->point_count = 0;
    node->nw = node->ne = node->sw = node->se = NULL;
    return node;
}

int contains_point(QuadTree *qt, double px, double py) {
    return px >= qt->x && px < qt->x + qt->width &&
           py >= qt->y && py < qt->y + qt->height;
}

void subdivide(QuadTree *qt) {
    double hw = qt->width / 2.0;
    double hh = qt->height / 2.0;
    
    qt->nw = create_quadtree(qt->x, qt->y, hw, hh);
    qt->ne = create_quadtree(qt->x + hw, qt->y, hw, hh);
    qt->sw = create_quadtree(qt->x, qt->y + hh, hw, hh);
    qt->se = create_quadtree(qt->x + hw, qt->y + hh, hw, hh);
}

int insert_point(QuadTree *qt, double x, double y) {
    if (!contains_point(qt, x, y)) return 0;
    
    if (qt->point_count < MAX_POINTS_PER_NODE && qt->nw == NULL) {
        Point *p = (Point*)malloc(sizeof(Point));
        p->x = x;
        p->y = y;
        p->next = qt->points;
        qt->points = p;
        qt->point_count++;
        return 1;
    }
    
    if (qt->nw == NULL) {
        subdivide(qt);
    }
    
    if (insert_point(qt->nw, x, y)) return 1;
    if (insert_point(qt->ne, x, y)) return 1;
    if (insert_point(qt->sw, x, y)) return 1;
    if (insert_point(qt->se, x, y)) return 1;
    
    return 0;
}

int range_query(QuadTree *qt, double qx, double qy, double qw, double qh, int *count) {
    if (qt->x > qx + qw || qt->x + qt->width < qx ||
        qt->y > qy + qh || qt->y + qt->height < qy) {
        return 0;
    }
    
    Point *p = qt->points;
    while (p) {
        if (p->x >= qx && p->x < qx + qw && p->y >= qy && p->y < qy + qh) {
            (*count)++;
        }
        p = p->next;
    }
    
    if (qt->nw) {
        range_query(qt->nw, qx, qy, qw, qh, count);
        range_query(qt->ne, qx, qy, qw, qh, count);
        range_query(qt->sw, qx, qy, qw, qh, count);
        range_query(qt->se, qx, qy, qw, qh, count);
    }
    
    return *count;
}

int main() {
    QuadTree *qt = create_quadtree(0.0, 0.0, 100.0, 100.0);
    
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    for (int i = 0; i < NUM_POINTS; i++) {
        seed = seed * 1103515245 + 12345;
        double x = ((seed & 0xFFFF) / (double)0xFFFF) * 100.0;
        seed = seed * 1103515245 + 12345;
        double y = ((seed & 0xFFFF) / (double)0xFFFF) * 100.0;
        
        insert_point(qt, x, y);
    }
    
    int total_found = 0;
    for (int q = 0; q < 100; q++) {
        seed = seed * 1103515245 + 12345;
        double qx = ((seed & 0xFFFF) / (double)0xFFFF) * 80.0;
        seed = seed * 1103515245 + 12345;
        double qy = ((seed & 0xFFFF) / (double)0xFFFF) * 80.0;
        
        int count = 0;
        range_query(qt, qx, qy, 20.0, 20.0, &count);
        total_found += count;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Quadtree: %d points, 100 range queries, %.6f seconds\n",
           NUM_POINTS, time_spent);
    printf("Total points found in ranges: %d\n", total_found);
    
    return 0;
}
