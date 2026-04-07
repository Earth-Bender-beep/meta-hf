// K-means clustering algorithm
// Unsupervised learning, iterative refinement
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_POINTS 1000
#define NUM_CLUSTERS 8
#define DIMENSIONS 5
#define MAX_ITERATIONS 100

typedef struct {
    double coords[DIMENSIONS];
    int cluster;
} Point;

typedef struct {
    double coords[DIMENSIONS];
    int count;
} Centroid;

double euclidean_distance(double *a, double *b) {
    double sum = 0.0;
    for (int i = 0; i < DIMENSIONS; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

void init_points(Point *points, unsigned int *seed) {
    for (int i = 0; i < NUM_POINTS; i++) {
        for (int j = 0; j < DIMENSIONS; j++) {
            *seed = *seed * 1103515245 + 12345;
            points[i].coords[j] = ((*seed) & 0xFFFF) / (double)0xFFFF * 100.0;
        }
        points[i].cluster = -1;
    }
}

void init_centroids(Centroid *centroids, Point *points, unsigned int *seed) {
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        *seed = *seed * 1103515245 + 12345;
        int idx = (*seed) % NUM_POINTS;
        
        for (int j = 0; j < DIMENSIONS; j++) {
            centroids[i].coords[j] = points[idx].coords[j];
        }
        centroids[i].count = 0;
    }
}

int assign_clusters(Point *points, Centroid *centroids) {
    int changes = 0;
    
    for (int i = 0; i < NUM_POINTS; i++) {
        double min_dist = 1e10;
        int nearest = 0;
        
        for (int j = 0; j < NUM_CLUSTERS; j++) {
            double dist = euclidean_distance(points[i].coords, centroids[j].coords);
            if (dist < min_dist) {
                min_dist = dist;
                nearest = j;
            }
        }
        
        if (points[i].cluster != nearest) {
            changes++;
            points[i].cluster = nearest;
        }
    }
    
    return changes;
}

void update_centroids(Point *points, Centroid *centroids) {
    // Reset centroids
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        for (int j = 0; j < DIMENSIONS; j++) {
            centroids[i].coords[j] = 0.0;
        }
        centroids[i].count = 0;
    }
    
    // Sum points in each cluster
    for (int i = 0; i < NUM_POINTS; i++) {
        int cluster = points[i].cluster;
        if (cluster >= 0) {
            for (int j = 0; j < DIMENSIONS; j++) {
                centroids[cluster].coords[j] += points[i].coords[j];
            }
            centroids[cluster].count++;
        }
    }
    
    // Calculate means
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        if (centroids[i].count > 0) {
            for (int j = 0; j < DIMENSIONS; j++) {
                centroids[i].coords[j] /= centroids[i].count;
            }
        }
    }
}

int kmeans(Point *points, Centroid *centroids, unsigned int *seed) {
    init_centroids(centroids, points, seed);
    
    int iterations = 0;
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        iterations++;
        
        int changes = assign_clusters(points, centroids);
        update_centroids(points, centroids);
        
        if (changes == 0) break;
    }
    
    return iterations;
}

int main() {
    unsigned int seed = 12345;
    Point *points = (Point*)malloc(NUM_POINTS * sizeof(Point));
    Centroid *centroids = (Centroid*)malloc(NUM_CLUSTERS * sizeof(Centroid));
    
    init_points(points, &seed);
    
    clock_t start = clock();
    int iterations = kmeans(points, centroids, &seed);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Count cluster sizes
    int cluster_sizes[NUM_CLUSTERS] = {0};
    for (int i = 0; i < NUM_POINTS; i++) {
        if (points[i].cluster >= 0) {
            cluster_sizes[points[i].cluster]++;
        }
    }
    
    printf("K-means: %d points, %d clusters, %.6f seconds\n",
           NUM_POINTS, NUM_CLUSTERS, time_spent);
    printf("Converged in %d iterations\n", iterations);
    printf("Cluster sizes: ");
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        printf("%d ", cluster_sizes[i]);
    }
    printf("\n");
    
    free(points);
    free(centroids);
    
    return 0;
}
