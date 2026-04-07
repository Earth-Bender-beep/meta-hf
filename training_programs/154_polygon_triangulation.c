// Polygon triangulation using ear clipping method
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_VERTICES 100

typedef struct {
    double x, y;
} Point;

typedef struct {
    int a, b, c;
} Triangle;

double cross_product(Point *o, Point *a, Point *b) {
    return (a->x - o->x) * (b->y - o->y) - (a->y - o->y) * (b->x - o->x);
}

int is_convex(Point *prev, Point *curr, Point *next) {
    return cross_product(prev, curr, next) > 0;
}

int point_in_triangle(Point *p, Point *a, Point *b, Point *c) {
    double d1 = cross_product(a, b, p);
    double d2 = cross_product(b, c, p);
    double d3 = cross_product(c, a, p);
    
    int has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    int has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    
    return !(has_neg && has_pos);
}

int is_ear(Point *polygon, int *indices, int n, int i) {
    int prev = (i - 1 + n) % n;
    int next = (i + 1) % n;
    
    if (!is_convex(&polygon[indices[prev]], &polygon[indices[i]], &polygon[indices[next]])) {
        return 0;
    }
    
    for (int j = 0; j < n; j++) {
        if (j == prev || j == i || j == next) continue;
        
        if (point_in_triangle(&polygon[indices[j]],
                             &polygon[indices[prev]],
                             &polygon[indices[i]],
                             &polygon[indices[next]])) {
            return 0;
        }
    }
    
    return 1;
}

int ear_clipping(Point *polygon, int n, Triangle *triangles) {
    int *indices = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }
    
    int num_triangles = 0;
    int count = n;
    
    while (count > 3) {
        int ear_found = 0;
        
        for (int i = 0; i < count; i++) {
            if (is_ear(polygon, indices, count, i)) {
                int prev = (i - 1 + count) % count;
                int next = (i + 1) % count;
                
                triangles[num_triangles].a = indices[prev];
                triangles[num_triangles].b = indices[i];
                triangles[num_triangles].c = indices[next];
                num_triangles++;
                
                for (int j = i; j < count - 1; j++) {
                    indices[j] = indices[j + 1];
                }
                count--;
                ear_found = 1;
                break;
            }
        }
        
        if (!ear_found) break;
    }
    
    if (count == 3) {
        triangles[num_triangles].a = indices[0];
        triangles[num_triangles].b = indices[1];
        triangles[num_triangles].c = indices[2];
        num_triangles++;
    }
    
    free(indices);
    return num_triangles;
}

int main() {
    Point polygon[50];
    Triangle triangles[50];
    
    int n = 30;
    unsigned int seed = 42;
    
    for (int i = 0; i < n; i++) {
        double angle = 2.0 * M_PI * i / n;
        seed = seed * 1103515245 + 12345;
        double radius = 10.0 + ((seed & 0xFF) / (double)0xFF) * 5.0;
        
        polygon[i].x = radius * cos(angle);
        polygon[i].y = radius * sin(angle);
    }
    
    clock_t start = clock();
    int num_triangles = ear_clipping(polygon, n, triangles);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Polygon triangulation: %d vertices, %d triangles, %.6f seconds\n",
           n, num_triangles, time_spent);
    
    return 0;
}
