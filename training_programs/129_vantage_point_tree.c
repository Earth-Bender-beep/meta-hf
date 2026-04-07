// Vantage Point Tree (VP-tree) for nearest neighbor search in metric spaces
// Efficient spatial data structure for high-dimensional spaces
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DIM 5
#define NUM_POINTS 5000
#define NUM_QUERIES 1000

typedef struct {
    double coords[DIM];
} Point;

typedef struct VPNode {
    Point point;
    double median_dist;
    struct VPNode *left;
    struct VPNode *right;
} VPNode;

double distance(Point *a, Point *b) {
    double sum = 0.0;
    for (int i = 0; i < DIM; i++) {
        double diff = a->coords[i] - b->coords[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

int compare_dist(const void *a, const void *b) {
    double *da = (double*)a;
    double *db = (double*)b;
    if (*da < *db) return -1;
    if (*da > *db) return 1;
    return 0;
}

VPNode* build_vp_tree(Point points[], int n) {
    if (n == 0) return NULL;
    
    VPNode *node = (VPNode*)malloc(sizeof(VPNode));
    node->point = points[0];
    
    if (n == 1) {
        node->left = NULL;
        node->right = NULL;
        node->median_dist = 0;
        return node;
    }
    
    // Calculate distances from vantage point
    double *distances = (double*)malloc((n - 1) * sizeof(double));
    for (int i = 1; i < n; i++) {
        distances[i - 1] = distance(&points[0], &points[i]);
    }
    
    // Find median distance
    qsort(distances, n - 1, sizeof(double), compare_dist);
    node->median_dist = distances[(n - 1) / 2];
    
    // Partition points
    Point *left_points = (Point*)malloc((n - 1) * sizeof(Point));
    Point *right_points = (Point*)malloc((n - 1) * sizeof(Point));
    int left_count = 0, right_count = 0;
    
    for (int i = 1; i < n; i++) {
        double dist = distance(&points[0], &points[i]);
        if (dist < node->median_dist) {
            left_points[left_count++] = points[i];
        } else {
            right_points[right_count++] = points[i];
        }
    }
    
    // Recursively build subtrees
    node->left = build_vp_tree(left_points, left_count);
    node->right = build_vp_tree(right_points, right_count);
    
    free(distances);
    free(left_points);
    free(right_points);
    
    return node;
}

void search_nearest(VPNode *node, Point *target, double *best_dist, Point *best_point) {
    if (!node) return;
    
    double dist = distance(&node->point, target);
    
    if (dist < *best_dist) {
        *best_dist = dist;
        *best_point = node->point;
    }
    
    if (dist < node->median_dist) {
        search_nearest(node->left, target, best_dist, best_point);
        if (dist + *best_dist >= node->median_dist) {
            search_nearest(node->right, target, best_dist, best_point);
        }
    } else {
        search_nearest(node->right, target, best_dist, best_point);
        if (dist - *best_dist < node->median_dist) {
            search_nearest(node->left, target, best_dist, best_point);
        }
    }
}

void free_vp_tree(VPNode *node) {
    if (!node) return;
    free_vp_tree(node->left);
    free_vp_tree(node->right);
    free(node);
}

void generate_points(Point points[], int n) {
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < DIM; j++) {
            seed = seed * 1103515245 + 12345;
            points[i].coords[j] = ((seed & 0xFFFF) / (double)0xFFFF) * 100.0;
        }
    }
}

int main() {
    Point *points = (Point*)malloc(NUM_POINTS * sizeof(Point));
    Point *queries = (Point*)malloc(NUM_QUERIES * sizeof(Point));
    
    generate_points(points, NUM_POINTS);
    generate_points(queries, NUM_QUERIES);
    
    clock_t start = clock();
    
    VPNode *tree = build_vp_tree(points, NUM_POINTS);
    
    double total_distance = 0.0;
    
    for (int i = 0; i < NUM_QUERIES; i++) {
        double best_dist = 1e100;
        Point best_point;
        search_nearest(tree, &queries[i], &best_dist, &best_point);
        total_distance += best_dist;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("VP-tree: %d points (dim=%d), %d queries, %.6f seconds\n",
           NUM_POINTS, DIM, NUM_QUERIES, time_spent);
    printf("Average nearest neighbor distance: %.4f\n", total_distance / NUM_QUERIES);
    
    free_vp_tree(tree);
    free(points);
    free(queries);
    
    return 0;
}
