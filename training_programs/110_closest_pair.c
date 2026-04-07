// Closest pair of points using divide and conquer
// O(n log n) computational geometry algorithm
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_POINTS 10000

typedef struct {
    double x, y;
} Point;

double dist(Point p1, Point p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

int compare_x(const void *a, const void *b) {
    Point *p1 = (Point*)a;
    Point *p2 = (Point*)b;
    if (p1->x < p2->x) return -1;
    if (p1->x > p2->x) return 1;
    return 0;
}

int compare_y(const void *a, const void *b) {
    Point *p1 = (Point*)a;
    Point *p2 = (Point*)b;
    if (p1->y < p2->y) return -1;
    if (p1->y > p2->y) return 1;
    return 0;
}

double brute_force(Point points[], int n) {
    double min_dist = 1e10;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double d = dist(points[i], points[j]);
            if (d < min_dist) {
                min_dist = d;
            }
        }
    }
    return min_dist;
}

double strip_closest(Point strip[], int size, double d) {
    double min_dist = d;
    
    qsort(strip, size, sizeof(Point), compare_y);
    
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size && (strip[j].y - strip[i].y) < min_dist; j++) {
            double dist_ij = dist(strip[i], strip[j]);
            if (dist_ij < min_dist) {
                min_dist = dist_ij;
            }
        }
    }
    
    return min_dist;
}

double closest_pair_rec(Point px[], Point py[], int n) {
    if (n <= 3) {
        return brute_force(px, n);
    }
    
    int mid = n / 2;
    Point mid_point = px[mid];
    
    Point *pyl = (Point*)malloc(mid * sizeof(Point));
    Point *pyr = (Point*)malloc((n - mid) * sizeof(Point));
    
    int li = 0, ri = 0;
    for (int i = 0; i < n; i++) {
        if (py[i].x <= mid_point.x && li < mid) {
            pyl[li++] = py[i];
        } else {
            pyr[ri++] = py[i];
        }
    }
    
    double dl = closest_pair_rec(px, pyl, mid);
    double dr = closest_pair_rec(px + mid, pyr, n - mid);
    
    double d = (dl < dr) ? dl : dr;
    
    Point *strip = (Point*)malloc(n * sizeof(Point));
    int strip_size = 0;
    
    for (int i = 0; i < n; i++) {
        if (fabs(py[i].x - mid_point.x) < d) {
            strip[strip_size++] = py[i];
        }
    }
    
    double strip_min = strip_closest(strip, strip_size, d);
    
    free(pyl);
    free(pyr);
    free(strip);
    
    return (d < strip_min) ? d : strip_min;
}

double closest_pair(Point points[], int n) {
    Point *px = (Point*)malloc(n * sizeof(Point));
    Point *py = (Point*)malloc(n * sizeof(Point));
    
    for (int i = 0; i < n; i++) {
        px[i] = points[i];
        py[i] = points[i];
    }
    
    qsort(px, n, sizeof(Point), compare_x);
    qsort(py, n, sizeof(Point), compare_y);
    
    double result = closest_pair_rec(px, py, n);
    
    free(px);
    free(py);
    
    return result;
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
    generate_points(points, NUM_POINTS);
    
    clock_t start = clock();
    double min_distance = closest_pair(points, NUM_POINTS);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Closest pair (divide & conquer): %d points, %.6f seconds\n",
           NUM_POINTS, time_spent);
    printf("Minimum distance: %.6f\n", min_distance);
    
    free(points);
    return 0;
}
