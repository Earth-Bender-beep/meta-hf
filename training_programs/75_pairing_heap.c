// Pairing heap - simplified fibonacci heap for priority queue
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_OPS 8000

typedef struct PairingNode {
    int key;
    struct PairingNode *child;
    struct PairingNode *sibling;
} PairingNode;

PairingNode* create_node(int key) {
    PairingNode *node = (PairingNode*)malloc(sizeof(PairingNode));
    node->key = key;
    node->child = NULL;
    node->sibling = NULL;
    return node;
}

PairingNode* meld(PairingNode *a, PairingNode *b) {
    if (!a) return b;
    if (!b) return a;
    
    if (a->key < b->key) {
        b->sibling = a->child;
        a->child = b;
        return a;
    } else {
        a->sibling = b->child;
        b->child = a;
        return b;
    }
}

PairingNode* insert(PairingNode *heap, int key) {
    return meld(heap, create_node(key));
}

PairingNode* merge_pairs(PairingNode *node) {
    if (!node || !node->sibling) {
        return node;
    }
    
    PairingNode *a = node;
    PairingNode *b = node->sibling;
    PairingNode *rest = b->sibling;
    
    a->sibling = NULL;
    b->sibling = NULL;
    
    return meld(meld(a, b), merge_pairs(rest));
}

PairingNode* delete_min(PairingNode *heap, int *min_val) {
    if (!heap) return NULL;
    
    *min_val = heap->key;
    PairingNode *child = heap->child;
    free(heap);
    
    return merge_pairs(child);
}

int main() {
    PairingNode *heap = NULL;
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    for (int i = 0; i < NUM_OPS; i++) {
        seed = seed * 1103515245 + 12345;
        heap = insert(heap, seed % 100000);
    }
    
    long long sum = 0;
    for (int i = 0; i < NUM_OPS / 2; i++) {
        int min_val;
        heap = delete_min(heap, &min_val);
        sum += min_val;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Pairing heap: %d operations, %.6f seconds\n", NUM_OPS, time_spent);
    printf("Sum: %lld\n", sum);
    
    return 0;
}
