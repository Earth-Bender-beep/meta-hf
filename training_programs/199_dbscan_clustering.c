// DBSCAN clustering algorithm for density-based clustering
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_POINTS 2000
#define DIM 2
#define EPSILON 5.0
#define MIN_POINTS 5

typedef struct {
    double coords[DIM];
    int cluster_id;
    int visited;
} Point;

double euclidean_distance(Point *a, Point *b) {
    double sum = 0.0;
    for (int i = 0; i < DIM; i++) {
        double diff = a->coords[i] - b->coords[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

void range_query(Point *points, int n, int point_idx, double eps, int *neighbors, int *count) {
    *count = 0;
    for (int i = 0; i < n; i++) {
        if (euclidean_distance(&points[point_idx], &points[i]) <= eps) {
            neighbors[(*count)++] = i;
        }
    }
}

void expand_cluster(Point *points, int n, int point_idx, int cluster_id, double eps, int min_pts) {
    int *neighbors = (int*)malloc(n * sizeof(int));
    int neighbor_count;
    
    range_query(points, n, point_idx, eps, neighbors, &neighbor_count);
    
    if (neighbor_count < min_pts) {
        points[point_idx].cluster_id = -1;
        free(neighbors);
        return;
    }
    
    points[point_idx].cluster_id = cluster_id;
    
    int *queue = (int*)malloc(n * sizeof(int));
    int queue_size = 0;
    
    for (int i = 0; i < neighbor_count; i++) {
        int neighbor_idx = neighbors[i];
        if (neighbor_idx != point_idx) {
            queue[queue_size++] = neighbor_idx;
        }
    }
    
    int queue_pos = 0;
    while (queue_pos < queue_size) {
        int current_idx = queue[queue_pos++];
        
        if (points[current_idx].visited) continue;
        points[current_idx].visited = 1;
        
        int current_neighbor_count;
        range_query(points, n, current_idx, eps, neighbors, &current_neighbor_count);
        
        if (current_neighbor_count >= min_pts) {
            for (int i = 0; i < current_neighbor_count; i++) {
                int neighbor_idx = neighbors[i];
                if (points[neighbor_idx].cluster_id == 0) {
                    queue[queue_size++] = neighbor_idx;
                }
            }
        }
        
        if (points[current_idx].cluster_id == 0) {
            points[current_idx].cluster_id = cluster_id;
        }
    }
    
    free(neighbors);
    free(queue);
}

int dbscan(Point *points, int n, double eps, int min_pts) {
    int cluster_id = 0;
    
    for (int i = 0; i < n; i++) {
        points[i].cluster_id = 0;
        points[i].visited = 0;
    }
    
    for (int i = 0; i < n; i++) {
        if (points[i].visited) continue;
        
        points[i].visited = 1;
        
        int *neighbors = (int*)malloc(n * sizeof(int));
        int neighbor_count;
        range_query(points, n, i, eps, neighbors, &neighbor_count);
        
        if (neighbor_count >= min_pts) {
            cluster_id++;
            expand_cluster(points, n, i, cluster_id, eps, min_pts);
        } else {
            points[i].cluster_id = -1;
        }
        
        free(neighbors);
    }
    
    return cluster_id;
}

int main() {
    int n = NUM_POINTS;
    Point *points = (Point*)malloc(n * sizeof(Point));
    
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < DIM; d++) {
            seed = seed * 1103515245 + 12345;
            points[i].coords[d] = ((seed & 0xFFFF) / (double)0xFFFF) * 100.0;
        }
    }
    
    clock_t start = clock();
    
    int num_clusters = dbscan(points, n, EPSILON, MIN_POINTS);
    
    int noise_points = 0;
    for (int i = 0; i < n; i++) {
        if (points[i].cluster_id == -1) {
            noise_points++;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("DBSCAN: %d points, eps=%.1f, min_pts=%d, %.6f seconds\n",
           n, EPSILON, MIN_POINTS, time_spent);
    printf("Clusters found: %d, Noise points: %d\n", num_clusters, noise_points);
    
    free(points);
    
    return 0;
}
