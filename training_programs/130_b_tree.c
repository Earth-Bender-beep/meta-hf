// B-tree implementation - self-balancing tree optimized for disk access
// Each node can have multiple keys and children
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define T 3  // Minimum degree (each node has at least T-1 keys)
#define NUM_OPERATIONS 5000

typedef struct BTreeNode {
    int *keys;
    struct BTreeNode **children;
    int num_keys;
    int is_leaf;
} BTreeNode;

typedef struct {
    BTreeNode *root;
} BTree;

BTreeNode* create_node(int is_leaf) {
    BTreeNode *node = (BTreeNode*)malloc(sizeof(BTreeNode));
    node->keys = (int*)malloc((2 * T - 1) * sizeof(int));
    node->children = (BTreeNode**)malloc(2 * T * sizeof(BTreeNode*));
    node->num_keys = 0;
    node->is_leaf = is_leaf;
    return node;
}

BTree* create_btree() {
    BTree *tree = (BTree*)malloc(sizeof(BTree));
    tree->root = create_node(1);
    return tree;
}

void split_child(BTreeNode *parent, int index) {
    BTreeNode *full_child = parent->children[index];
    BTreeNode *new_child = create_node(full_child->is_leaf);
    
    new_child->num_keys = T - 1;
    
    // Copy last T-1 keys to new node
    for (int i = 0; i < T - 1; i++) {
        new_child->keys[i] = full_child->keys[i + T];
    }
    
    // Copy last T children if not leaf
    if (!full_child->is_leaf) {
        for (int i = 0; i < T; i++) {
            new_child->children[i] = full_child->children[i + T];
        }
    }
    
    full_child->num_keys = T - 1;
    
    // Shift parent's children
    for (int i = parent->num_keys; i > index; i--) {
        parent->children[i + 1] = parent->children[i];
    }
    parent->children[index + 1] = new_child;
    
    // Shift parent's keys
    for (int i = parent->num_keys - 1; i >= index; i--) {
        parent->keys[i + 1] = parent->keys[i];
    }
    parent->keys[index] = full_child->keys[T - 1];
    parent->num_keys++;
}

void insert_non_full(BTreeNode *node, int key) {
    int i = node->num_keys - 1;
    
    if (node->is_leaf) {
        // Shift keys and insert
        while (i >= 0 && key < node->keys[i]) {
            node->keys[i + 1] = node->keys[i];
            i--;
        }
        node->keys[i + 1] = key;
        node->num_keys++;
    } else {
        // Find child to insert into
        while (i >= 0 && key < node->keys[i]) {
            i--;
        }
        i++;
        
        if (node->children[i]->num_keys == 2 * T - 1) {
            split_child(node, i);
            if (key > node->keys[i]) {
                i++;
            }
        }
        insert_non_full(node->children[i], key);
    }
}

void btree_insert(BTree *tree, int key) {
    BTreeNode *root = tree->root;
    
    if (root->num_keys == 2 * T - 1) {
        BTreeNode *new_root = create_node(0);
        new_root->children[0] = root;
        split_child(new_root, 0);
        tree->root = new_root;
        insert_non_full(new_root, key);
    } else {
        insert_non_full(root, key);
    }
}

BTreeNode* search(BTreeNode *node, int key) {
    int i = 0;
    while (i < node->num_keys && key > node->keys[i]) {
        i++;
    }
    
    if (i < node->num_keys && key == node->keys[i]) {
        return node;
    }
    
    if (node->is_leaf) {
        return NULL;
    }
    
    return search(node->children[i], key);
}

int get_height(BTreeNode *node) {
    if (!node || node->is_leaf) return 1;
    return 1 + get_height(node->children[0]);
}

void free_btree(BTreeNode *node) {
    if (!node) return;
    
    if (!node->is_leaf) {
        for (int i = 0; i <= node->num_keys; i++) {
            free_btree(node->children[i]);
        }
    }
    
    free(node->keys);
    free(node->children);
    free(node);
}

int main() {
    BTree *tree = create_btree();
    
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    // Insert operations
    for (int i = 0; i < NUM_OPERATIONS; i++) {
        seed = seed * 1103515245 + 12345;
        int key = seed % 50000;
        btree_insert(tree, key);
    }
    
    // Search operations
    int found = 0;
    for (int i = 0; i < NUM_OPERATIONS / 2; i++) {
        seed = seed * 1103515245 + 12345;
        int key = seed % 50000;
        if (search(tree->root, key)) {
            found++;
        }
    }
    
    int height = get_height(tree->root);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("B-tree (degree=%d): %d operations, %.6f seconds\n", T, NUM_OPERATIONS, time_spent);
    printf("Tree height: %d, Found: %d\n", height, found);
    
    free_btree(tree->root);
    free(tree);
    
    return 0;
}
