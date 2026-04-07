// PageRank algorithm for graph ranking
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_NODES 100
#define MAX_ITERATIONS 50
#define DAMPING_FACTOR 0.85
#define TOLERANCE 1e-6

void compute_pagerank(double **adj_matrix, int n, double *pagerank, double damping) {
    double *new_rank = (double*)malloc(n * sizeof(double));
    int *out_degree = (int*)calloc(n, sizeof(int));
    
    for (int i = 0; i < n; i++) {
        pagerank[i] = 1.0 / n;
        for (int j = 0; j < n; j++) {
            if (adj_matrix[i][j] > 0) {
                out_degree[i]++;
            }
        }
    }
    
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        double diff = 0.0;
        
        for (int i = 0; i < n; i++) {
            new_rank[i] = (1.0 - damping) / n;
            
            for (int j = 0; j < n; j++) {
                if (adj_matrix[j][i] > 0 && out_degree[j] > 0) {
                    new_rank[i] += damping * pagerank[j] / out_degree[j];
                }
            }
            
            diff += fabs(new_rank[i] - pagerank[i]);
        }
        
        for (int i = 0; i < n; i++) {
            pagerank[i] = new_rank[i];
        }
        
        if (diff < TOLERANCE) {
            break;
        }
    }
    
    free(new_rank);
    free(out_degree);
}

int main() {
    int n = NUM_NODES;
    double **adj_matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        adj_matrix[i] = (double*)calloc(n, sizeof(double));
    }
    
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                seed = seed * 1103515245 + 12345;
                if ((seed & 0xFF) < 20) {
                    adj_matrix[i][j] = 1.0;
                }
            }
        }
    }
    
    double *pagerank = (double*)malloc(n * sizeof(double));
    
    clock_t start = clock();
    
    compute_pagerank(adj_matrix, n, pagerank, DAMPING_FACTOR);
    
    double sum = 0.0;
    int max_idx = 0;
    for (int i = 0; i < n; i++) {
        sum += pagerank[i];
        if (pagerank[i] > pagerank[max_idx]) {
            max_idx = i;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("PageRank: %d nodes, %.6f seconds\n", n, time_spent);
    printf("Sum of ranks: %.6f, Max rank: %.6f (node %d)\n",
           sum, pagerank[max_idx], max_idx);
    
    for (int i = 0; i < n; i++) free(adj_matrix[i]);
    free(adj_matrix);
    free(pagerank);
    
    return 0;
}
