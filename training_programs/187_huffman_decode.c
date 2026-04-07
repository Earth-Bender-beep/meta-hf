// Huffman decoding algorithm
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_SYMBOLS 256

typedef struct HuffNode {
    unsigned char symbol;
    int is_leaf;
    struct HuffNode *left, *right;
} HuffNode;

HuffNode* create_node(unsigned char symbol, int is_leaf) {
    HuffNode *node = (HuffNode*)malloc(sizeof(HuffNode));
    node->symbol = symbol;
    node->is_leaf = is_leaf;
    node->left = node->right = NULL;
    return node;
}

HuffNode* build_simple_tree() {
    HuffNode *root = create_node(0, 0);
    root->left = create_node('a', 1);
    root->right = create_node(0, 0);
    root->right->left = create_node('b', 1);
    root->right->right = create_node('c', 1);
    return root;
}

int huffman_decode(HuffNode *root, unsigned char *encoded, int num_bits, unsigned char *decoded) {
    int decoded_len = 0;
    HuffNode *current = root;
    
    for (int i = 0; i < num_bits; i++) {
        int byte_idx = i / 8;
        int bit_idx = 7 - (i % 8);
        int bit = (encoded[byte_idx] >> bit_idx) & 1;
        
        current = bit ? current->right : current->left;
        
        if (current->is_leaf) {
            decoded[decoded_len++] = current->symbol;
            current = root;
        }
    }
    
    return decoded_len;
}

int main() {
    HuffNode *tree = build_simple_tree();
    
    unsigned char encoded[] = {0b01101100, 0b11000000};
    unsigned char *decoded = (unsigned char*)malloc(1000);
    
    clock_t start = clock();
    
    long long total_decoded = 0;
    for (int test = 0; test < 100000; test++) {
        int decoded_len = huffman_decode(tree, encoded, 10, decoded);
        total_decoded += decoded_len;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Huffman decode: 100000 iterations, %.6f seconds\n", time_spent);
    printf("Total decoded: %lld symbols\n", total_decoded);
    
    free(decoded);
    return 0;
}
