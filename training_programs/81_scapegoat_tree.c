// Scapegoat tree - self-balancing BST without storing balance info
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_OPS 5000
#define ALPHA 0.7

typedef struct ScapegoatNode {
    int key;
    struct ScapegoatNode *left;
    struct ScapegoatNode *right;
} ScapegoatNode;

ScapegoatNode* create_node(int key) {
    ScapegoatNode *node = (ScapegoatNode*)malloc(sizeof(ScapegoatNode));
    node->key = key;
    node->left = NULL;
    node->right = NULL;
    return node;
}

int get_size(ScapegoatNode *node) {
    if (!node) return 0;
    return 1 + get_size(node->left) + get_size(node->right);
}

void flatten_tree(ScapegoatNode *node, ScapegoatNode **arr, int *idx) {
    if (!node) return;
    flatten_tree(node->left, arr, idx);
    arr[(*idx)++] = node;
    flatten_tree(node->right, arr, idx);
}

ScapegoatNode* build_balanced(ScapegoatNode **arr, int start, int end) {
    if (start > end) return NULL;
    
    int mid = (start + end) / 2;
    ScapegoatNode *root = arr[mid];
    root->left = build_balanced(arr, start, mid - 1);
    root->right = build_balanced(arr, mid + 1, end);
    
    return root;
}

ScapegoatNode* rebuild(ScapegoatNode *node) {
    int size = get_size(node);
    ScapegoatNode **arr = (ScapegoatNode**)malloc(size * sizeof(ScapegoatNode*));
    int idx = 0;
    
    flatten_tree(node, arr, &idx);
    ScapegoatNode *result = build_balanced(arr, 0, size - 1);
    
    free(arr);
    return result;
}

ScapegoatNode* insert(ScapegoatNode *root, int key, int *depth, int max_depth) {
    if (!root) {
        *depth = 0;
        return create_node(key);
    }
    
    if (key < root->key) {
        root->left = insert(root->left, key, depth, max_depth);
    } else {
        root->right = insert(root->right, key, depth, max_depth);
    }
    
    (*depth)++;
    
    if (*depth > max_depth) {
        int left_size = get_size(root->left);
        int right_size = get_size(root->right);
        
        if (left_size > ALPHA * (left_size + right_size + 1)) {
            root = rebuild(root);
        }
    }
    
    return root;
}

int main() {
    ScapegoatNode *root = NULL;
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    for (int i = 0; i < NUM_OPS; i++) {
        seed = seed * 1103515245 + 12345;
        int depth = 0;
        int max_depth = (int)(log(i + 1) / log(1.0 / ALPHA));
        root = insert(root, seed % 100000, &depth, max_depth);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Scapegoat tree: %d operations, %.6f seconds\n", NUM_OPS, time_spent);
    
    return 0;
}
