// Graham scan algorithm for computing convex hull
// O(n log n) computational geometry algorithm
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_POINTS 5000

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

int compare_points(const void *a, const void *b) {
    Point *p1 = (Point*)a;
    Point *p2 = (Point*)b;
    
    double cp = cross_product(pivot, *p1, *p2);
    
    if (cp == 0) {
        // Collinear points - choose closer one
        return distance_sq(pivot, *p1) < distance_sq(pivot, *p2) ? -1 : 1;
    }
    
    return cp < 0 ? 1 : -1;
}

int graham_scan(Point points[], int n, Point hull[]) {
    if (n < 3) return n;
    
    // Find bottom-most point (or left-most if tie)
    int min_idx = 0;
    for (int i = 1; i < n; i++) {
        if (points[i].y < points[min_idx].y ||
            (points[i].y == points[min_idx].y && points[i].x < points[min_idx].x)) {
            min_idx = i;
        }
    }
    
    // Swap with first position
    Point temp = points[0];
    points[0] = points[min_idx];
    points[min_idx] = temp;
    
    pivot = points[0];
    
    // Sort points by polar angle
    qsort(points + 1, n - 1, sizeof(Point), compare_points);
    
    // Build convex hull
    hull[0] = points[0];
    hull[1] = points[1];
    hull[2] = points[2];
    int hull_size = 3;
    
    for (int i = 3; i < n; i++) {
        // Remove points that make clockwise turn
        while (hull_size >= 2 && 
               cross_product(hull[hull_size-2], hull[hull_size-1], points[i]) <= 0) {
            hull_size--;
        }
        hull[hull_size++] = points[i];
    }
    
    return hull_size;
}

void generate_points(Point points[], int n) {
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        points[i].x = ((seed & 0xFFFF) / (double)0xFFFF) * 1000.0;
        seed = seed * 1103515245 + 12345;
        points[i].y = ((seed & 0xFFFF) / (double)0xFFFF) * 1000.0;
    }
}

int main() {
    Point *points = (Point*)malloc(NUM_POINTS * sizeof(Point));
    Point *hull = (Point*)malloc(NUM_POINTS * sizeof(Point));
    
    generate_points(points, NUM_POINTS);
    
    clock_t start = clock();
    
    int total_hull_points = 0;
    for (int iter = 0; iter < 50; iter++) {
        Point *temp_points = (Point*)malloc(NUM_POINTS * sizeof(Point));
        for (int i = 0; i < NUM_POINTS; i++) {
            temp_points[i] = points[i];
        }
        
        int hull_size = graham_scan(temp_points, NUM_POINTS, hull);
        total_hull_points += hull_size;
        
        free(temp_points);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Graham scan (convex hull): %d points, 50 iterations, %.6f seconds\n",
           NUM_POINTS, time_spent);
    printf("Average hull size: %.2f\n", total_hull_points / 50.0);
    
    free(points);
    free(hull);
    
    return 0;
}
