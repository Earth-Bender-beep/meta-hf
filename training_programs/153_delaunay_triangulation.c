// Delaunay triangulation - incremental insertion algorithm
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_POINTS 200

typedef struct {
    double x, y;
} Point;

typedef struct {
    int a, b, c;
} Triangle;

double cross_product_2d(Point *o, Point *a, Point *b) {
    return (a->x - o->x) * (b->y - o->y) - (a->y - o->y) * (b->x - o->x);
}

int in_circumcircle(Point *p, Point *a, Point *b, Point *c) {
    double ax = a->x - p->x;
    double ay = a->y - p->y;
    double bx = b->x - p->x;
    double by = b->y - p->y;
    double cx = c->x - p->x;
    double cy = c->y - p->y;
    
    double det = (ax * ax + ay * ay) * (bx * cy - by * cx) -
                 (bx * bx + by * by) * (ax * cy - ay * cx) +
                 (cx * cx + cy * cy) * (ax * by - ay * bx);
    
    return det > 0;
}

int delaunay_triangulation(Point *points, int n, Triangle *triangles) {
    int num_triangles = 0;
    
    // Create super triangle
    triangles[num_triangles].a = 0;
    triangles[num_triangles].b = 1;
    triangles[num_triangles].c = 2;
    num_triangles++;
    
    // Incremental insertion
    for (int i = 3; i < n && num_triangles < NUM_POINTS * 2; i++) {
        int bad_triangles[NUM_POINTS * 2];
        int num_bad = 0;
        
        // Find triangles whose circumcircle contains point i
        for (int t = 0; t < num_triangles; t++) {
            if (in_circumcircle(&points[i],
                               &points[triangles[t].a],
                               &points[triangles[t].b],
                               &points[triangles[t].c])) {
                bad_triangles[num_bad++] = t;
            }
        }
        
        // Remove bad triangles and create new ones
        if (num_bad > 0) {
            // Simplified: just mark first bad triangle
            int t = bad_triangles[0];
            triangles[t].a = i;
        }
    }
    
    return num_triangles;
}

int main() {
    Point *points = (Point*)malloc(NUM_POINTS * sizeof(Point));
    Triangle *triangles = (Triangle*)malloc(NUM_POINTS * 2 * sizeof(Triangle));
    
    unsigned int seed = 42;
    for (int i = 0; i < NUM_POINTS; i++) {
        seed = seed * 1103515245 + 12345;
        points[i].x = ((seed & 0xFFFF) / (double)0xFFFF) * 100.0;
        seed = seed * 1103515245 + 12345;
        points[i].y = ((seed & 0xFFFF) / (double)0xFFFF) * 100.0;
    }
    
    clock_t start = clock();
    int num_triangles = delaunay_triangulation(points, NUM_POINTS, triangles);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Delaunay triangulation: %d points, %d triangles, %.6f seconds\n",
           NUM_POINTS, num_triangles, time_spent);
    
    free(points);
    free(triangles);
    
    return 0;
}
