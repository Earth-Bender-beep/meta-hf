#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef enum { RED, BLACK } Color;

typedef struct Node {
    int data;
    Color color;
    struct Node *left, *right, *parent;
} Node;

Node* createNode(int data) {
    Node* node = (Node*)malloc(sizeof(Node));
    node->data = data;
    node->color = RED;
    node->left = node->right = node->parent = NULL;
    return node;
}

Node* bst_insert(Node* root, Node* pt) {
    if (root == NULL)
        return pt;
    
    if (pt->data < root->data) {
        root->left = bst_insert(root->left, pt);
        root->left->parent = root;
    } else if (pt->data > root->data) {
        root->right = bst_insert(root->right, pt);
        root->right->parent = root;
    }
    
    return root;
}

void freeTree(Node* root) {
    if (root != NULL) {
        freeTree(root->left);
        freeTree(root->right);
        free(root);
    }
}

int main() {
    int n = 30000;
    Node* root = NULL;
    
    srand(42);
    clock_t start = clock();
    
    for (int i = 0; i < n; i++) {
        Node* newNode = createNode(rand() % 100000);
        root = bst_insert(root, newNode);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Red-Black tree: %d insertions in %.6f seconds\n", n, time_spent);
    
    freeTree(root);
    return 0;
}
