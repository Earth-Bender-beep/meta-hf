// K-means clustering with k-means++ initialization
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_POINTS 5000
#define NUM_CLUSTERS 10
#define MAX_ITERATIONS 50
#define DIM 2

double distance_squared(double *a, double *b, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

void kmeans_plusplus_init(double **points, int n, double **centroids, int k, int dim, unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    int first = (*seed) % n;
    for (int d = 0; d < dim; d++) {
        centroids[0][d] = points[first][d];
    }
    
    for (int c = 1; c < k; c++) {
        double *min_dist = (double*)malloc(n * sizeof(double));
        double sum_dist = 0.0;
        
        for (int i = 0; i < n; i++) {
            min_dist[i] = INFINITY;
            for (int j = 0; j < c; j++) {
                double dist = distance_squared(points[i], centroids[j], dim);
                if (dist < min_dist[i]) {
                    min_dist[i] = dist;
                }
            }
            sum_dist += min_dist[i];
        }
        
        *seed = *seed * 1103515245 + 12345;
        double r = ((*seed & 0x7FFFFFFF) / (double)0x7FFFFFFF) * sum_dist;
        
        double cumulative = 0.0;
        for (int i = 0; i < n; i++) {
            cumulative += min_dist[i];
            if (cumulative >= r) {
                for (int d = 0; d < dim; d++) {
                    centroids[c][d] = points[i][d];
                }
                break;
            }
        }
        
        free(min_dist);
    }
}

double kmeans(double **points, int n, double **centroids, int k, int dim, int *assignments) {
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        int changed = 0;
        
        for (int i = 0; i < n; i++) {
            double min_dist = INFINITY;
            int best_cluster = 0;
            
            for (int c = 0; c < k; c++) {
                double dist = distance_squared(points[i], centroids[c], dim);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            
            if (assignments[i] != best_cluster) {
                assignments[i] = best_cluster;
                changed = 1;
            }
        }
        
        if (!changed) break;
        
        int *cluster_counts = (int*)calloc(k, sizeof(int));
        for (int c = 0; c < k; c++) {
            for (int d = 0; d < dim; d++) {
                centroids[c][d] = 0.0;
            }
        }
        
        for (int i = 0; i < n; i++) {
            int c = assignments[i];
            cluster_counts[c]++;
            for (int d = 0; d < dim; d++) {
                centroids[c][d] += points[i][d];
            }
        }
        
        for (int c = 0; c < k; c++) {
            if (cluster_counts[c] > 0) {
                for (int d = 0; d < dim; d++) {
                    centroids[c][d] /= cluster_counts[c];
                }
            }
        }
        
        free(cluster_counts);
    }
    
    double inertia = 0.0;
    for (int i = 0; i < n; i++) {
        inertia += distance_squared(points[i], centroids[assignments[i]], DIM);
    }
    
    return inertia;
}

int main() {
    int n = NUM_POINTS;
    int k = NUM_CLUSTERS;
    
    double **points = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        points[i] = (double*)malloc(DIM * sizeof(double));
    }
    
    double **centroids = (double**)malloc(k * sizeof(double*));
    for (int i = 0; i < k; i++) {
        centroids[i] = (double*)malloc(DIM * sizeof(double));
    }
    
    int *assignments = (int*)malloc(n * sizeof(int));
    
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < DIM; d++) {
            seed = seed * 1103515245 + 12345;
            points[i][d] = ((seed & 0xFFFF) / (double)0xFFFF) * 100.0;
        }
        assignments[i] = -1;
    }
    
    clock_t start = clock();
    
    kmeans_plusplus_init(points, n, centroids, k, DIM, &seed);
    double inertia = kmeans(points, n, centroids, k, DIM, assignments);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("K-means++: %d points, %d clusters, %.6f seconds\n", n, k, time_spent);
    printf("Inertia: %.2f\n", inertia);
    
    for (int i = 0; i < n; i++) free(points[i]);
    free(points);
    for (int i = 0; i < k; i++) free(centroids[i]);
    free(centroids);
    free(assignments);
    
    return 0;
}
