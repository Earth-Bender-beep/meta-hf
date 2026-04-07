// Treap (Tree + Heap) - randomized binary search tree
// BST by key, heap by priority - provides balanced tree with high probability
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_OPERATIONS 10000

typedef struct TreapNode {
    int key;
    int priority;
    struct TreapNode *left;
    struct TreapNode *right;
} TreapNode;

unsigned int rand_seed = 42;

int my_rand() {
    rand_seed = rand_seed * 1103515245 + 12345;
    return (rand_seed >> 16) & 0x7FFF;
}

TreapNode* create_node(int key) {
    TreapNode *node = (TreapNode*)malloc(sizeof(TreapNode));
    node->key = key;
    node->priority = my_rand();
    node->left = NULL;
    node->right = NULL;
    return node;
}

TreapNode* rotate_right(TreapNode *root) {
    TreapNode *new_root = root->left;
    root->left = new_root->right;
    new_root->right = root;
    return new_root;
}

TreapNode* rotate_left(TreapNode *root) {
    TreapNode *new_root = root->right;
    root->right = new_root->left;
    new_root->left = root;
    return new_root;
}

TreapNode* insert(TreapNode *root, int key) {
    if (!root) {
        return create_node(key);
    }
    
    if (key < root->key) {
        root->left = insert(root->left, key);
        
        // Maintain heap property
        if (root->left->priority > root->priority) {
            root = rotate_right(root);
        }
    } else if (key > root->key) {
        root->right = insert(root->right, key);
        
        // Maintain heap property
        if (root->right->priority > root->priority) {
            root = rotate_left(root);
        }
    }
    
    return root;
}

TreapNode* delete_node(TreapNode *root, int key) {
    if (!root) return NULL;
    
    if (key < root->key) {
        root->left = delete_node(root->left, key);
    } else if (key > root->key) {
        root->right = delete_node(root->right, key);
    } else {
        // Found the node to delete
        if (!root->left && !root->right) {
            free(root);
            return NULL;
        } else if (!root->left) {
            TreapNode *temp = root->right;
            free(root);
            return temp;
        } else if (!root->right) {
            TreapNode *temp = root->left;
            free(root);
            return temp;
        } else {
            // Both children exist - rotate based on priority
            if (root->left->priority > root->right->priority) {
                root = rotate_right(root);
                root->right = delete_node(root->right, key);
            } else {
                root = rotate_left(root);
                root->left = delete_node(root->left, key);
            }
        }
    }
    
    return root;
}

TreapNode* search(TreapNode *root, int key) {
    if (!root || root->key == key) {
        return root;
    }
    
    if (key < root->key) {
        return search(root->left, key);
    } else {
        return search(root->right, key);
    }
}

int get_height(TreapNode *root) {
    if (!root) return 0;
    
    int left_height = get_height(root->left);
    int right_height = get_height(root->right);
    
    return 1 + (left_height > right_height ? left_height : right_height);
}

void free_treap(TreapNode *root) {
    if (!root) return;
    free_treap(root->left);
    free_treap(root->right);
    free(root);
}

int main() {
    TreapNode *root = NULL;
    
    clock_t start = clock();
    
    // Insert operations
    for (int i = 0; i < NUM_OPERATIONS; i++) {
        root = insert(root, my_rand() % 50000);
    }
    
    // Search operations
    int found = 0;
    for (int i = 0; i < NUM_OPERATIONS / 2; i++) {
        if (search(root, my_rand() % 50000)) {
            found++;
        }
    }
    
    int height = get_height(root);
    
    // Delete operations
    for (int i = 0; i < NUM_OPERATIONS / 3; i++) {
        root = delete_node(root, my_rand() % 50000);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Treap: %d operations, %.6f seconds\n", NUM_OPERATIONS, time_spent);
    printf("Tree height: %d, Found: %d\n", height, found);
    
    free_treap(root);
    
    return 0;
}
