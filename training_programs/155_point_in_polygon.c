// Point in polygon test using ray casting algorithm
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_VERTICES 100
#define NUM_TESTS 10000

typedef struct {
    double x, y;
} Point;

int point_in_polygon(Point *point, Point *polygon, int n) {
    int inside = 0;
    
    for (int i = 0, j = n - 1; i < n; j = i++) {
        double xi = polygon[i].x, yi = polygon[i].y;
        double xj = polygon[j].x, yj = polygon[j].y;
        
        int intersect = ((yi > point->y) != (yj > point->y)) &&
                       (point->x < (xj - xi) * (point->y - yi) / (yj - yi) + xi);
        
        if (intersect) {
            inside = !inside;
        }
    }
    
    return inside;
}

int winding_number(Point *point, Point *polygon, int n) {
    int wn = 0;
    
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        
        if (polygon[i].y <= point->y) {
            if (polygon[j].y > point->y) {
                double cross = (polygon[j].x - polygon[i].x) * (point->y - polygon[i].y) -
                              (point->x - polygon[i].x) * (polygon[j].y - polygon[i].y);
                if (cross > 0) {
                    wn++;
                }
            }
        } else {
            if (polygon[j].y <= point->y) {
                double cross = (polygon[j].x - polygon[i].x) * (point->y - polygon[i].y) -
                              (point->x - polygon[i].x) * (polygon[j].y - polygon[i].y);
                if (cross < 0) {
                    wn--;
                }
            }
        }
    }
    
    return wn != 0;
}

int main() {
    Point *polygon = (Point*)malloc(NUM_VERTICES * sizeof(Point));
    
    unsigned int seed = 42;
    for (int i = 0; i < NUM_VERTICES; i++) {
        seed = seed * 1103515245 + 12345;
        polygon[i].x = ((seed & 0xFFFF) / (double)0xFFFF) * 100.0;
        seed = seed * 1103515245 + 12345;
        polygon[i].y = ((seed & 0xFFFF) / (double)0xFFFF) * 100.0;
    }
    
    clock_t start = clock();
    
    int inside_ray = 0;
    int inside_winding = 0;
    
    for (int test = 0; test < NUM_TESTS; test++) {
        seed = seed * 1103515245 + 12345;
        Point p = {
            ((seed & 0xFFFF) / (double)0xFFFF) * 100.0,
            ((seed >> 16) & 0xFFFF) / (double)0xFFFF * 100.0
        };
        
        if (point_in_polygon(&p, polygon, NUM_VERTICES)) {
            inside_ray++;
        }
        
        if (winding_number(&p, polygon, NUM_VERTICES)) {
            inside_winding++;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Point in polygon: %d vertices, %d tests, %.6f seconds\n",
           NUM_VERTICES, NUM_TESTS, time_spent);
    printf("Ray casting: %d inside, Winding number: %d inside\n",
           inside_ray, inside_winding);
    
    free(polygon);
    
    return 0;
}
