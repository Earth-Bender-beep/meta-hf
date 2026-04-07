// Jarvis march (Gift wrapping) algorithm for convex hull
// Computational geometry, different from Graham scan
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_POINTS 2000

typedef struct {
    int x;
    int y;
} Point;

int orientation(Point p, Point q, Point r) {
    long long val = (long long)(q.y - p.y) * (r.x - q.x) - 
                    (long long)(q.x - p.x) * (r.y - q.y);
    
    if (val == 0) return 0;
    return (val > 0) ? 1 : 2;
}

int jarvis_march(Point *points, int n, Point *hull) {
    if (n < 3) return 0;
    
    // Find leftmost point
    int leftmost = 0;
    for (int i = 1; i < n; i++) {
        if (points[i].x < points[leftmost].x) {
            leftmost = i;
        }
    }
    
    int hull_size = 0;
    int p = leftmost;
    
    do {
        hull[hull_size++] = points[p];
        
        int q = (p + 1) % n;
        for (int i = 0; i < n; i++) {
            if (orientation(points[p], points[i], points[q]) == 2) {
                q = i;
            }
        }
        
        p = q;
    } while (p != leftmost);
    
    return hull_size;
}

void generate_points(Point *points, int n) {
    for (int i = 0; i < n; i++) {
        points[i].x = (i * 17 + 13) % 1000;
        points[i].y = (i * 23 + 7) % 1000;
    }
}

int main() {
    Point *points = (Point*)malloc(N_POINTS * sizeof(Point));
    Point *hull = (Point*)malloc(N_POINTS * sizeof(Point));
    
    generate_points(points, N_POINTS);
    
    clock_t start = clock();
    int hull_size = jarvis_march(points, N_POINTS, hull);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Jarvis march: %d points, %.6f seconds\n", N_POINTS, time_spent);
    printf("Convex hull size: %d\n", hull_size);
    
    free(points);
    free(hull);
    return 0;
}
