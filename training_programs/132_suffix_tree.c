// Simplified suffix tree construction (Ukkonen's algorithm simulation)
// For substring search and pattern matching
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_CHAR 256
#define TEXT_SIZE 5000
#define NUM_PATTERNS 500

typedef struct SuffixTreeNode {
    struct SuffixTreeNode *children[MAX_CHAR];
    int start;
    int *end;
    int suffix_index;
} SuffixTreeNode;

typedef struct {
    char *text;
    SuffixTreeNode *root;
    int size;
} SuffixTree;

SuffixTreeNode* create_node(int start, int *end) {
    SuffixTreeNode *node = (SuffixTreeNode*)malloc(sizeof(SuffixTreeNode));
    for (int i = 0; i < MAX_CHAR; i++) {
        node->children[i] = NULL;
    }
    node->start = start;
    node->end = end;
    node->suffix_index = -1;
    return node;
}

SuffixTree* create_suffix_tree(const char *text) {
    SuffixTree *tree = (SuffixTree*)malloc(sizeof(SuffixTree));
    int len = strlen(text);
    tree->text = (char*)malloc(len + 1);
    strcpy(tree->text, text);
    tree->size = len;
    
    int *leaf_end = (int*)malloc(sizeof(int));
    *leaf_end = -1;
    
    tree->root = create_node(-1, leaf_end);
    
    // Simplified construction - add suffixes directly
    for (int i = 0; i < len; i++) {
        SuffixTreeNode *curr = tree->root;
        
        for (int j = i; j < len; j++) {
            unsigned char c = text[j];
            
            if (!curr->children[c]) {
                int *end = (int*)malloc(sizeof(int));
                *end = len - 1;
                curr->children[c] = create_node(j, end);
                curr->children[c]->suffix_index = i;
                break;
            }
            curr = curr->children[c];
        }
    }
    
    return tree;
}

int search_pattern(SuffixTree *tree, const char *pattern) {
    int len = strlen(pattern);
    SuffixTreeNode *curr = tree->root;
    
    for (int i = 0; i < len; i++) {
        unsigned char c = pattern[i];
        if (!curr->children[c]) {
            return 0;
        }
        curr = curr->children[c];
    }
    
    return 1;
}

void free_suffix_tree(SuffixTreeNode *node) {
    if (!node) return;
    
    for (int i = 0; i < MAX_CHAR; i++) {
        if (node->children[i]) {
            free_suffix_tree(node->children[i]);
        }
    }
    
    if (node->end) {
        free(node->end);
    }
    free(node);
}

void generate_text(char *text, int size) {
    unsigned int seed = 42;
    for (int i = 0; i < size - 1; i++) {
        seed = seed * 1103515245 + 12345;
        text[i] = 'a' + (seed % 4);  // Small alphabet
    }
    text[size - 1] = '\0';
}

int main() {
    char *text = (char*)malloc(TEXT_SIZE);
    generate_text(text, TEXT_SIZE);
    
    clock_t start = clock();
    
    SuffixTree *tree = create_suffix_tree(text);
    
    int matches = 0;
    for (int i = 0; i < NUM_PATTERNS; i++) {
        // Search for substrings from the text
        int pos = (i * 17) % (TEXT_SIZE - 10);
        char pattern[11];
        strncpy(pattern, text + pos, 10);
        pattern[10] = '\0';
        
        if (search_pattern(tree, pattern)) {
            matches++;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Suffix tree: text=%d, %d searches, %.6f seconds\n",
           TEXT_SIZE, NUM_PATTERNS, time_spent);
    printf("Matches found: %d\n", matches);
    
    free_suffix_tree(tree->root);
    free(tree->text);
    free(tree);
    free(text);
    
    return 0;
}
