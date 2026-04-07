// Leftist heap - heap-ordered binary tree with leftist property
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_OPS 7000

typedef struct LeftistNode {
    int key;
    int npl;
    struct LeftistNode *left;
    struct LeftistNode *right;
} LeftistNode;

LeftistNode* create_node(int key) {
    LeftistNode *node = (LeftistNode*)malloc(sizeof(LeftistNode));
    node->key = key;
    node->npl = 0;
    node->left = NULL;
    node->right = NULL;
    return node;
}

LeftistNode* merge(LeftistNode *h1, LeftistNode *h2) {
    if (!h1) return h2;
    if (!h2) return h1;
    
    if (h1->key > h2->key) {
        LeftistNode *temp = h1;
        h1 = h2;
        h2 = temp;
    }
    
    h1->right = merge(h1->right, h2);
    
    int left_npl = h1->left ? h1->left->npl : -1;
    int right_npl = h1->right ? h1->right->npl : -1;
    
    if (left_npl < right_npl) {
        LeftistNode *temp = h1->left;
        h1->left = h1->right;
        h1->right = temp;
    }
    
    h1->npl = (h1->right ? h1->right->npl : -1) + 1;
    
    return h1;
}

LeftistNode* insert(LeftistNode *heap, int key) {
    return merge(heap, create_node(key));
}

int main() {
    LeftistNode *heap = NULL;
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    for (int i = 0; i < NUM_OPS; i++) {
        seed = seed * 1103515245 + 12345;
        heap = insert(heap, seed % 100000);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Leftist heap: %d operations, %.6f seconds\n", NUM_OPS, time_spent);
    
    return 0;
}
