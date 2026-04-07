// Interval tree for overlapping interval queries
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_INTERVALS 3000

typedef struct Interval {
    int low;
    int high;
} Interval;

typedef struct IntervalNode {
    Interval interval;
    int max;
    struct IntervalNode *left;
    struct IntervalNode *right;
} IntervalNode;

IntervalNode* create_node(Interval interval) {
    IntervalNode *node = (IntervalNode*)malloc(sizeof(IntervalNode));
    node->interval = interval;
    node->max = interval.high;
    node->left = NULL;
    node->right = NULL;
    return node;
}

int max(int a, int b) {
    return (a > b) ? a : b;
}

IntervalNode* insert(IntervalNode *root, Interval interval) {
    if (!root) return create_node(interval);
    
    int low = root->interval.low;
    
    if (interval.low < low) {
        root->left = insert(root->left, interval);
    } else {
        root->right = insert(root->right, interval);
    }
    
    if (root->max < interval.high) {
        root->max = interval.high;
    }
    
    return root;
}

int overlaps(Interval a, Interval b) {
    return a.low <= b.high && b.low <= a.high;
}

Interval* search_overlap(IntervalNode *root, Interval interval) {
    if (!root) return NULL;
    
    if (overlaps(root->interval, interval)) {
        return &root->interval;
    }
    
    if (root->left && root->left->max >= interval.low) {
        return search_overlap(root->left, interval);
    }
    
    return search_overlap(root->right, interval);
}

int main() {
    IntervalNode *root = NULL;
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    for (int i = 0; i < NUM_INTERVALS; i++) {
        seed = seed * 1103515245 + 12345;
        int low = seed % 10000;
        seed = seed * 1103515245 + 12345;
        int high = low + (seed % 100) + 1;
        
        Interval interval = {low, high};
        root = insert(root, interval);
    }
    
    int overlaps_found = 0;
    for (int i = 0; i < 1000; i++) {
        seed = seed * 1103515245 + 12345;
        Interval query = {seed % 10000, (seed % 10000) + 50};
        
        if (search_overlap(root, query)) {
            overlaps_found++;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Interval tree: %d intervals, %.6f seconds\n", NUM_INTERVALS, time_spent);
    printf("Overlaps found: %d/1000\n", overlaps_found);
    
    return 0;
}
