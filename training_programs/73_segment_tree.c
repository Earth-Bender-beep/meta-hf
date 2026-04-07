// Segment tree for range queries and updates
// Tree structure, logarithmic operations
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000

typedef struct {
    int *tree;
    int n;
} SegmentTree;

int min(int a, int b) {
    return (a < b) ? a : b;
}

SegmentTree* create_segment_tree(int *arr, int n) {
    SegmentTree *st = (SegmentTree*)malloc(sizeof(SegmentTree));
    st->n = n;
    st->tree = (int*)malloc(4 * n * sizeof(int));
    return st;
}

void build_tree(SegmentTree *st, int *arr, int node, int start, int end) {
    if (start == end) {
        st->tree[node] = arr[start];
    } else {
        int mid = (start + end) / 2;
        build_tree(st, arr, 2*node, start, mid);
        build_tree(st, arr, 2*node+1, mid+1, end);
        st->tree[node] = min(st->tree[2*node], st->tree[2*node+1]);
    }
}

void update(SegmentTree *st, int node, int start, int end, int idx, int val) {
    if (start == end) {
        st->tree[node] = val;
    } else {
        int mid = (start + end) / 2;
        if (idx <= mid) {
            update(st, 2*node, start, mid, idx, val);
        } else {
            update(st, 2*node+1, mid+1, end, idx, val);
        }
        st->tree[node] = min(st->tree[2*node], st->tree[2*node+1]);
    }
}

int query(SegmentTree *st, int node, int start, int end, int l, int r) {
    if (r < start || end < l) {
        return 1000000000;
    }
    
    if (l <= start && end <= r) {
        return st->tree[node];
    }
    
    int mid = (start + end) / 2;
    int left_min = query(st, 2*node, start, mid, l, r);
    int right_min = query(st, 2*node+1, mid+1, end, l, r);
    
    return min(left_min, right_min);
}

void free_segment_tree(SegmentTree *st) {
    free(st->tree);
    free(st);
}

int main() {
    int *arr = (int*)malloc(N * sizeof(int));
    
    for (int i = 0; i < N; i++) {
        arr[i] = i % 1000;
    }
    
    clock_t start = clock();
    
    SegmentTree *st = create_segment_tree(arr, N);
    build_tree(st, arr, 1, 0, N-1);
    
    // Perform queries
    int sum = 0;
    for (int i = 0; i < 1000; i++) {
        int l = i % (N/2);
        int r = l + (N/4);
        sum += query(st, 1, 0, N-1, l, r);
    }
    
    // Perform updates
    for (int i = 0; i < 500; i++) {
        update(st, 1, 0, N-1, i*2, i);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Segment tree: %d elements, %.6f seconds\n", N, time_spent);
    printf("Query sum: %d\n", sum);
    
    free_segment_tree(st);
    free(arr);
    return 0;
}
