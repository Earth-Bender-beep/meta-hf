// Fenwick tree (Binary Indexed Tree) for prefix sums
// Bit manipulation based tree structure
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 15000

typedef struct {
    int *tree;
    int n;
} FenwickTree;

FenwickTree* create_fenwick_tree(int n) {
    FenwickTree *ft = (FenwickTree*)malloc(sizeof(FenwickTree));
    ft->n = n;
    ft->tree = (int*)calloc(n + 1, sizeof(int));
    return ft;
}

void update(FenwickTree *ft, int idx, int delta) {
    while (idx <= ft->n) {
        ft->tree[idx] += delta;
        idx += idx & (-idx);  // Add last set bit
    }
}

int query(FenwickTree *ft, int idx) {
    int sum = 0;
    while (idx > 0) {
        sum += ft->tree[idx];
        idx -= idx & (-idx);  // Remove last set bit
    }
    return sum;
}

int range_query(FenwickTree *ft, int left, int right) {
    return query(ft, right) - query(ft, left - 1);
}

void build_tree(FenwickTree *ft, int *arr, int n) {
    for (int i = 1; i <= n; i++) {
        update(ft, i, arr[i-1]);
    }
}

void free_fenwick_tree(FenwickTree *ft) {
    free(ft->tree);
    free(ft);
}

int main() {
    int *arr = (int*)malloc(N * sizeof(int));
    
    for (int i = 0; i < N; i++) {
        arr[i] = (i % 100) + 1;
    }
    
    clock_t start = clock();
    
    FenwickTree *ft = create_fenwick_tree(N);
    build_tree(ft, arr, N);
    
    // Perform range queries
    long long sum = 0;
    for (int i = 0; i < 5000; i++) {
        int left = 1 + (i % (N/2));
        int right = left + (N/4);
        if (right > N) right = N;
        sum += range_query(ft, left, right);
    }
    
    // Perform updates
    for (int i = 0; i < 2000; i++) {
        update(ft, 1 + (i % N), i % 50);
    }
    
    // More queries after updates
    for (int i = 0; i < 3000; i++) {
        sum += query(ft, N - i % 1000);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Fenwick tree: %d elements, %.6f seconds\n", N, time_spent);
    printf("Total sum: %lld\n", sum);
    
    free_fenwick_tree(ft);
    free(arr);
    return 0;
}
