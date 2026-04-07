// Splay tree - self-adjusting binary search tree
// Recently accessed elements are moved to root via splaying
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_OPERATIONS 10000

typedef struct SplayNode {
    int key;
    struct SplayNode *left;
    struct SplayNode *right;
} SplayNode;

SplayNode* create_node(int key) {
    SplayNode *node = (SplayNode*)malloc(sizeof(SplayNode));
    node->key = key;
    node->left = NULL;
    node->right = NULL;
    return node;
}

SplayNode* right_rotate(SplayNode *x) {
    SplayNode *y = x->left;
    x->left = y->right;
    y->right = x;
    return y;
}

SplayNode* left_rotate(SplayNode *x) {
    SplayNode *y = x->right;
    x->right = y->left;
    y->left = x;
    return y;
}

SplayNode* splay(SplayNode *root, int key) {
    if (!root || root->key == key) {
        return root;
    }
    
    // Key is in left subtree
    if (key < root->key) {
        if (!root->left) return root;
        
        // Zig-Zig (Left Left)
        if (key < root->left->key) {
            root->left->left = splay(root->left->left, key);
            root = right_rotate(root);
        }
        // Zig-Zag (Left Right)
        else if (key > root->left->key) {
            root->left->right = splay(root->left->right, key);
            if (root->left->right) {
                root->left = left_rotate(root->left);
            }
        }
        
        return root->left ? right_rotate(root) : root;
    }
    // Key is in right subtree
    else {
        if (!root->right) return root;
        
        // Zig-Zag (Right Left)
        if (key < root->right->key) {
            root->right->left = splay(root->right->left, key);
            if (root->right->left) {
                root->right = right_rotate(root->right);
            }
        }
        // Zig-Zig (Right Right)
        else if (key > root->right->key) {
            root->right->right = splay(root->right->right, key);
            root = left_rotate(root);
        }
        
        return root->right ? left_rotate(root) : root;
    }
}

SplayNode* insert(SplayNode *root, int key) {
    if (!root) return create_node(key);
    
    root = splay(root, key);
    
    if (root->key == key) return root;
    
    SplayNode *new_node = create_node(key);
    
    if (key < root->key) {
        new_node->right = root;
        new_node->left = root->left;
        root->left = NULL;
    } else {
        new_node->left = root;
        new_node->right = root->right;
        root->right = NULL;
    }
    
    return new_node;
}

SplayNode* search(SplayNode *root, int key) {
    return splay(root, key);
}

SplayNode* delete_node(SplayNode *root, int key) {
    if (!root) return NULL;
    
    root = splay(root, key);
    
    if (root->key != key) return root;
    
    SplayNode *temp;
    if (!root->left) {
        temp = root->right;
        free(root);
        return temp;
    }
    
    temp = root;
    root = splay(root->left, key);
    root->right = temp->right;
    free(temp);
    
    return root;
}

void free_tree(SplayNode *root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int get_height(SplayNode *root) {
    if (!root) return 0;
    int left_height = get_height(root->left);
    int right_height = get_height(root->right);
    return 1 + (left_height > right_height ? left_height : right_height);
}

int main() {
    SplayNode *root = NULL;
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    // Insert operations
    for (int i = 0; i < NUM_OPERATIONS; i++) {
        seed = seed * 1103515245 + 12345;
        int key = seed % 50000;
        root = insert(root, key);
    }
    
    // Search operations (creates locality)
    int found = 0;
    for (int i = 0; i < NUM_OPERATIONS / 2; i++) {
        seed = seed * 1103515245 + 12345;
        int key = seed % 50000;
        root = search(root, key);
        if (root && root->key == key) {
            found++;
        }
    }
    
    int height = get_height(root);
    
    // Delete operations
    for (int i = 0; i < NUM_OPERATIONS / 4; i++) {
        seed = seed * 1103515245 + 12345;
        int key = seed % 50000;
        root = delete_node(root, key);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Splay tree: %d operations, %.6f seconds\n", NUM_OPERATIONS, time_spent);
    printf("Tree height: %d, Found: %d\n", height, found);
    
    free_tree(root);
    
    return 0;
}
