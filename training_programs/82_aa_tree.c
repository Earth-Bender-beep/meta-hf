// AA Tree - simplified red-black tree using levels
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_OPS 6000

typedef struct AANode {
    int key;
    int level;
    struct AANode *left;
    struct AANode *right;
} AANode;

AANode* create_node(int key) {
    AANode *node = (AANode*)malloc(sizeof(AANode));
    node->key = key;
    node->level = 1;
    node->left = NULL;
    node->right = NULL;
    return node;
}

AANode* skew(AANode *node) {
    if (!node || !node->left) return node;
    
    if (node->left->level == node->level) {
        AANode *left = node->left;
        node->left = left->right;
        left->right = node;
        return left;
    }
    
    return node;
}

AANode* split(AANode *node) {
    if (!node || !node->right || !node->right->right) return node;
    
    if (node->level == node->right->right->level) {
        AANode *right = node->right;
        node->right = right->left;
        right->left = node;
        right->level++;
        return right;
    }
    
    return node;
}

AANode* insert(AANode *node, int key) {
    if (!node) return create_node(key);
    
    if (key < node->key) {
        node->left = insert(node->left, key);
    } else if (key > node->key) {
        node->right = insert(node->right, key);
    }
    
    node = skew(node);
    node = split(node);
    
    return node;
}

int main() {
    AANode *root = NULL;
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    for (int i = 0; i < NUM_OPS; i++) {
        seed = seed * 1103515245 + 12345;
        root = insert(root, seed % 100000);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("AA Tree: %d operations, %.6f seconds\n", NUM_OPS, time_spent);
    
    return 0;
}
