// Binomial heap - collection of binomial trees for priority queue
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_OPS 5000

typedef struct BinomialNode {
    int key;
    int degree;
    struct BinomialNode *parent;
    struct BinomialNode *child;
    struct BinomialNode *sibling;
} BinomialNode;

BinomialNode* create_node(int key) {
    BinomialNode *node = (BinomialNode*)malloc(sizeof(BinomialNode));
    node->key = key;
    node->degree = 0;
    node->parent = NULL;
    node->child = NULL;
    node->sibling = NULL;
    return node;
}

BinomialNode* merge_trees(BinomialNode *b1, BinomialNode *b2) {
    if (b1->key > b2->key) {
        BinomialNode *temp = b1;
        b1 = b2;
        b2 = temp;
    }
    
    b2->parent = b1;
    b2->sibling = b1->child;
    b1->child = b2;
    b1->degree++;
    
    return b1;
}

BinomialNode* union_heaps(BinomialNode *h1, BinomialNode *h2) {
    if (!h1) return h2;
    if (!h2) return h1;
    
    BinomialNode *head = NULL;
    BinomialNode **pos = &head;
    
    while (h1 && h2) {
        if (h1->degree <= h2->degree) {
            *pos = h1;
            h1 = h1->sibling;
        } else {
            *pos = h2;
            h2 = h2->sibling;
        }
        pos = &(*pos)->sibling;
    }
    
    *pos = h1 ? h1 : h2;
    
    return head;
}

BinomialNode* insert(BinomialNode *heap, int key) {
    BinomialNode *node = create_node(key);
    return union_heaps(heap, node);
}

int main() {
    BinomialNode *heap = NULL;
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    for (int i = 0; i < NUM_OPS; i++) {
        seed = seed * 1103515245 + 12345;
        heap = insert(heap, seed % 100000);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Binomial heap: %d insertions, %.6f seconds\n", NUM_OPS, time_spent);
    
    return 0;
}
