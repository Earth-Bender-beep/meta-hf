// Convex hull using Graham scan algorithm
// Sorting + geometric predicates, moderate branching
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_POINTS 1000

typedef struct {
    double x, y;
} Point;

Point pivot;

double cross_product(Point o, Point a, Point b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

double distance_sq(Point a, Point b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

int compare_polar(const void *va, const void *vb) {
    Point *a = (Point*)va;
    Point *b = (Point*)vb;
    
    double cross = cross_product(pivot, *a, *b);
    if (fabs(cross) < 1e-9) {
        return (distance_sq(pivot, *a) < distance_sq(pivot, *b)) ? -1 : 1;
    }
    return (cross > 0) ? -1 : 1;
}

int graham_scan(Point *points, int n, Point *hull) {
    if (n < 3) return 0;
    
    // Find lowest point
    int min_idx = 0;
    for (int i = 1; i < n; i++) {
        if (points[i].y < points[min_idx].y || 
            (points[i].y == points[min_idx].y && points[i].x < points[min_idx].x)) {
            min_idx = i;
        }
    }
    
    // Swap to first position
    Point temp = points[0];
    points[0] = points[min_idx];
    points[min_idx] = temp;
    
    pivot = points[0];
    
    // Sort by polar angle
    qsort(points + 1, n - 1, sizeof(Point), compare_polar);
    
    // Build hull
    hull[0] = points[0];
    hull[1] = points[1];
    hull[2] = points[2];
    int hull_size = 3;
    
    for (int i = 3; i < n; i++) {
        while (hull_size >= 2 && 
               cross_product(hull[hull_size-2], hull[hull_size-1], points[i]) <= 0) {
            hull_size--;
        }
        hull[hull_size++] = points[i];
    }
    
    return hull_size;
}

void generate_points(Point *points, int n) {
    for (int i = 0; i < n; i++) {
        double angle = 2.0 * M_PI * i / n;
        double radius = 100.0 + (i % 50);
        points[i].x = radius * cos(angle) + (i % 20 - 10);
        points[i].y = radius * sin(angle) + (i % 15 - 7);
    }
}

int main() {
    Point *points = (Point*)malloc(N_POINTS * sizeof(Point));
    Point *hull = (Point*)malloc(N_POINTS * sizeof(Point));
    
    generate_points(points, N_POINTS);
    
    clock_t start = clock();
    int hull_size = graham_scan(points, N_POINTS, hull);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Convex hull: %d points -> %d hull points, %.6f seconds\n",
           N_POINTS, hull_size, time_spent);
    
    free(points);
    free(hull);
    return 0;
}
