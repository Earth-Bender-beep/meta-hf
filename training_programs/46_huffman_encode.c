// Huffman coding for lossless compression
// Tree construction, priority queue, bit packing
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ALPHABET_SIZE 256
#define DATA_SIZE 10000

typedef struct Node {
    unsigned char symbol;
    int frequency;
    struct Node *left, *right;
} Node;

typedef struct {
    Node **data;
    int size;
    int capacity;
} MinHeap;

Node* create_node(unsigned char symbol, int freq) {
    Node *node = (Node*)malloc(sizeof(Node));
    node->symbol = symbol;
    node->frequency = freq;
    node->left = node->right = NULL;
    return node;
}

MinHeap* create_heap(int capacity) {
    MinHeap *heap = (MinHeap*)malloc(sizeof(MinHeap));
    heap->data = (Node**)malloc(capacity * sizeof(Node*));
    heap->size = 0;
    heap->capacity = capacity;
    return heap;
}

void swap_nodes(Node **a, Node **b) {
    Node *temp = *a;
    *a = *b;
    *b = temp;
}

void heapify_down(MinHeap *heap, int idx) {
    int smallest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
    
    if (left < heap->size && heap->data[left]->frequency < heap->data[smallest]->frequency)
        smallest = left;
    
    if (right < heap->size && heap->data[right]->frequency < heap->data[smallest]->frequency)
        smallest = right;
    
    if (smallest != idx) {
        swap_nodes(&heap->data[idx], &heap->data[smallest]);
        heapify_down(heap, smallest);
    }
}

void heapify_up(MinHeap *heap, int idx) {
    int parent = (idx - 1) / 2;
    if (idx > 0 && heap->data[idx]->frequency < heap->data[parent]->frequency) {
        swap_nodes(&heap->data[idx], &heap->data[parent]);
        heapify_up(heap, parent);
    }
}

void insert_heap(MinHeap *heap, Node *node) {
    heap->data[heap->size] = node;
    heapify_up(heap, heap->size);
    heap->size++;
}

Node* extract_min(MinHeap *heap) {
    Node *min = heap->data[0];
    heap->data[0] = heap->data[heap->size - 1];
    heap->size--;
    heapify_down(heap, 0);
    return min;
}

Node* build_huffman_tree(int *frequencies) {
    MinHeap *heap = create_heap(ALPHABET_SIZE);
    
    for (int i = 0; i < ALPHABET_SIZE; i++) {
        if (frequencies[i] > 0) {
            insert_heap(heap, create_node(i, frequencies[i]));
        }
    }
    
    while (heap->size > 1) {
        Node *left = extract_min(heap);
        Node *right = extract_min(heap);
        
        Node *parent = create_node(0, left->frequency + right->frequency);
        parent->left = left;
        parent->right = right;
        
        insert_heap(heap, parent);
    }
    
    Node *root = extract_min(heap);
    free(heap->data);
    free(heap);
    return root;
}

void generate_codes(Node *root, char *code, int depth, char codes[ALPHABET_SIZE][32]) {
    if (!root) return;
    
    if (!root->left && !root->right) {
        code[depth] = '\0';
        strcpy(codes[root->symbol], code);
        return;
    }
    
    if (root->left) {
        code[depth] = '0';
        generate_codes(root->left, code, depth + 1, codes);
    }
    
    if (root->right) {
        code[depth] = '1';
        generate_codes(root->right, code, depth + 1, codes);
    }
}

void free_tree(Node *root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

int main() {
    unsigned char *data = (unsigned char*)malloc(DATA_SIZE);
    int frequencies[ALPHABET_SIZE] = {0};
    
    // Generate test data with varying frequencies
    for (int i = 0; i < DATA_SIZE; i++) {
        data[i] = (i % 100 < 50) ? 'A' + (i % 5) : 'A' + (i % 26);
        frequencies[data[i]]++;
    }
    
    clock_t start = clock();
    
    Node *root = build_huffman_tree(frequencies);
    
    char codes[ALPHABET_SIZE][32] = {0};
    char code[32];
    generate_codes(root, code, 0, codes);
    
    // Calculate compressed size
    int compressed_bits = 0;
    for (int i = 0; i < DATA_SIZE; i++) {
        compressed_bits += strlen(codes[data[i]]);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Huffman encoding: %d bytes -> %d bits (%.2f%% compression), %.6f seconds\n",
           DATA_SIZE, compressed_bits, 100.0 * compressed_bits / (DATA_SIZE * 8), time_spent);
    
    free_tree(root);
    free(data);
    return 0;
}
