// Binary Search Tree with various operations
// Tree structure, recursive operations, traversals
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_OPERATIONS 6000

typedef struct BSTNode {
    int key;
    struct BSTNode *left;
    struct BSTNode *right;
} BSTNode;

BSTNode* create_bst_node(int key) {
    BSTNode *node = (BSTNode*)malloc(sizeof(BSTNode));
    node->key = key;
    node->left = NULL;
    node->right = NULL;
    return node;
}

BSTNode* bst_insert(BSTNode *root, int key) {
    if (!root) return create_bst_node(key);
    
    if (key < root->key) {
        root->left = bst_insert(root->left, key);
    } else if (key > root->key) {
        root->right = bst_insert(root->right, key);
    }
    
    return root;
}

BSTNode* bst_search(BSTNode *root, int key) {
    if (!root || root->key == key) return root;
    
    if (key < root->key) {
        return bst_search(root->left, key);
    } else {
        return bst_search(root->right, key);
    }
}

BSTNode* find_min(BSTNode *root) {
    while (root && root->left) {
        root = root->left;
    }
    return root;
}

BSTNode* bst_delete(BSTNode *root, int key) {
    if (!root) return root;
    
    if (key < root->key) {
        root->left = bst_delete(root->left, key);
    } else if (key > root->key) {
        root->right = bst_delete(root->right, key);
    } else {
        if (!root->left) {
            BSTNode *temp = root->right;
            free(root);
            return temp;
        } else if (!root->right) {
            BSTNode *temp = root->left;
            free(root);
            return temp;
        }
        
        BSTNode *temp = find_min(root->right);
        root->key = temp->key;
        root->right = bst_delete(root->right, temp->key);
    }
    
    return root;
}

void inorder_traversal(BSTNode *root, int *count) {
    if (!root) return;
    inorder_traversal(root->left, count);
    (*count)++;
    inorder_traversal(root->right, count);
}

void free_bst(BSTNode *root) {
    if (!root) return;
    free_bst(root->left);
    free_bst(root->right);
    free(root);
}

int main() {
    BSTNode *root = NULL;
    
    clock_t start = clock();
    
    // Insert operations
    for (int i = 0; i < N_OPERATIONS; i++) {
        root = bst_insert(root, (i * 19 + 7) % 10000);
    }
    
    // Search operations
    int found = 0;
    for (int i = 0; i < N_OPERATIONS; i++) {
        if (bst_search(root, i % 10000)) found++;
    }
    
    // Traversal
    int node_count = 0;
    inorder_traversal(root, &node_count);
    
    // Delete operations
    for (int i = 0; i < N_OPERATIONS / 3; i++) {
        root = bst_delete(root, (i * 11) % 10000);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("BST: %d operations, %.6f seconds\n", N_OPERATIONS, time_spent);
    printf("Found: %d, Nodes: %d\n", found, node_count);
    
    free_bst(root);
    return 0;
}
