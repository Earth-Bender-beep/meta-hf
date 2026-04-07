// Voronoi diagram computation - partitions plane based on proximity to sites
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_SITES 500
#define GRID_SIZE 1000
#define NUM_QUERIES 10000

typedef struct {
    double x;
    double y;
} Point;

typedef struct {
    Point site;
    int id;
} VoronoiCell;

double distance(Point *p1, Point *p2) {
    double dx = p1->x - p2->x;
    double dy = p1->y - p2->y;
    return sqrt(dx * dx + dy * dy);
}

int find_nearest_site(Point *query, VoronoiCell *cells, int num_cells) {
    int nearest = 0;
    double min_dist = distance(query, &cells[0].site);
    
    for (int i = 1; i < num_cells; i++) {
        double dist = distance(query, &cells[i].site);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = i;
        }
    }
    
    return nearest;
}

void lloyds_relaxation(VoronoiCell *cells, int num_cells, int iterations) {
    for (int iter = 0; iter < iterations; iter++) {
        double *sum_x = (double*)calloc(num_cells, sizeof(double));
        double *sum_y = (double*)calloc(num_cells, sizeof(double));
        int *count = (int*)calloc(num_cells, sizeof(int));
        
        for (int i = 0; i < GRID_SIZE; i += 10) {
            for (int j = 0; j < GRID_SIZE; j += 10) {
                Point p = {i, j};
                int nearest = find_nearest_site(&p, cells, num_cells);
                
                sum_x[nearest] += p.x;
                sum_y[nearest] += p.y;
                count[nearest]++;
            }
        }
        
        for (int i = 0; i < num_cells; i++) {
            if (count[i] > 0) {
                cells[i].site.x = sum_x[i] / count[i];
                cells[i].site.y = sum_y[i] / count[i];
            }
        }
        
        free(sum_x);
        free(sum_y);
        free(count);
    }
}

int main() {
    VoronoiCell *cells = (VoronoiCell*)malloc(NUM_SITES * sizeof(VoronoiCell));
    
    unsigned int seed = 42;
    
    for (int i = 0; i < NUM_SITES; i++) {
        seed = seed * 1103515245 + 12345;
        cells[i].site.x = (seed & 0xFFFF) / (double)0xFFFF * GRID_SIZE;
        seed = seed * 1103515245 + 12345;
        cells[i].site.y = (seed & 0xFFFF) / (double)0xFFFF * GRID_SIZE;
        cells[i].id = i;
    }
    
    clock_t start = clock();
    
    lloyds_relaxation(cells, NUM_SITES, 3);
    
    int query_results[NUM_SITES] = {0};
    for (int q = 0; q < NUM_QUERIES; q++) {
        seed = seed * 1103515245 + 12345;
        Point query = {
            (seed & 0xFFFF) / (double)0xFFFF * GRID_SIZE,
            ((seed >> 16) & 0xFFFF) / (double)0xFFFF * GRID_SIZE
        };
        
        int nearest = find_nearest_site(&query, cells, NUM_SITES);
        query_results[nearest]++;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Voronoi diagram: %d sites, %d queries, %.6f seconds\n",
           NUM_SITES, NUM_QUERIES, time_spent);
    
    free(cells);
    
    return 0;
}
