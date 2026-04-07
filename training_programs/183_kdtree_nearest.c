// K-D tree for nearest neighbor search
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_POINTS 10000
#define K 2

typedef struct KDNode {
    double point[K];
    struct KDNode *left, *right;
} KDNode;

KDNode* create_node(double *point) {
    KDNode *node = (KDNode*)malloc(sizeof(KDNode));
    for (int i = 0; i < K; i++) {
        node->point[i] = point[i];
    }
    node->left = node->right = NULL;
    return node;
}

KDNode* build_kdtree(double points[][K], int start, int end, int depth) {
    if (start > end) return NULL;
    
    int axis = depth % K;
    int mid = (start + end) / 2;
    
    // Simple selection instead of full sort
    for (int i = start; i <= end; i++) {
        int min_idx = i;
        for (int j = i + 1; j <= end; j++) {
            if (points[j][axis] < points[min_idx][axis]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            for (int k = 0; k < K; k++) {
                double temp = points[i][k];
                points[i][k] = points[min_idx][k];
                points[min_idx][k] = temp;
            }
        }
    }
    
    KDNode *node = create_node(points[mid]);
    node->left = build_kdtree(points, start, mid - 1, depth + 1);
    node->right = build_kdtree(points, mid + 1, end, depth + 1);
    
    return node;
}

double distance_squared(double *a, double *b) {
    double sum = 0;
    for (int i = 0; i < K; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

void nearest_neighbor(KDNode *root, double *query, int depth,
                     double *best_point, double *best_dist) {
    if (!root) return;
    
    double dist = distance_squared(root->point, query);
    if (dist < *best_dist) {
        *best_dist = dist;
        for (int i = 0; i < K; i++) {
            best_point[i] = root->point[i];
        }
    }
    
    int axis = depth % K;
    double diff = query[axis] - root->point[axis];
    
    KDNode *near = (diff < 0) ? root->left : root->right;
    KDNode *far = (diff < 0) ? root->right : root->left;
    
    nearest_neighbor(near, query, depth + 1, best_point, best_dist);
    
    if (diff * diff < *best_dist) {
        nearest_neighbor(far, query, depth + 1, best_point, best_dist);
    }
}

int main() {
    double (*points)[K] = malloc(NUM_POINTS * sizeof(*points));
    
    unsigned int seed = 42;
    for (int i = 0; i < NUM_POINTS; i++) {
        for (int j = 0; j < K; j++) {
            seed = seed * 1103515245 + 12345;
            points[i][j] = ((seed & 0xFFFF) / (double)0xFFFF) * 100.0;
        }
    }
    
    clock_t start = clock();
    
    KDNode *root = build_kdtree(points, 0, NUM_POINTS - 1, 0);
    
    double total_dist = 0.0;
    for (int q = 0; q < 100; q++) {
        double query[K];
        for (int j = 0; j < K; j++) {
            seed = seed * 1103515245 + 12345;
            query[j] = ((seed & 0xFFFF) / (double)0xFFFF) * 100.0;
        }
        
        double best_point[K];
        double best_dist = 1e30;
        nearest_neighbor(root, query, 0, best_point, &best_dist);
        total_dist += sqrt(best_dist);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("K-D tree: %d points, 100 queries, %.6f seconds\n",
           NUM_POINTS, time_spent);
    printf("Average nearest neighbor distance: %.6f\n", total_dist / 100);
    
    free(points);
    return 0;
}
