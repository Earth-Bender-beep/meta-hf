#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ALPHABET_SIZE 26

typedef struct TrieNode {
    struct TrieNode *children[ALPHABET_SIZE];
    int isEndOfWord;
} TrieNode;

TrieNode* createNode() {
    TrieNode* node = (TrieNode*)malloc(sizeof(TrieNode));
    node->isEndOfWord = 0;
    for (int i = 0; i < ALPHABET_SIZE; i++)
        node->children[i] = NULL;
    return node;
}

void insert(TrieNode *root, const char *key) {
    TrieNode *curr = root;
    while (*key) {
        int index = *key - 'a';
        if (!curr->children[index])
            curr->children[index] = createNode();
        curr = curr->children[index];
        key++;
    }
    curr->isEndOfWord = 1;
}

int search(TrieNode *root, const char *key) {
    TrieNode *curr = root;
    while (*key) {
        int index = *key - 'a';
        if (!curr->children[index])
            return 0;
        curr = curr->children[index];
        key++;
    }
    return (curr != NULL && curr->isEndOfWord);
}

void freeTrie(TrieNode* root) {
    if (root) {
        for (int i = 0; i < ALPHABET_SIZE; i++)
            if (root->children[i])
                freeTrie(root->children[i]);
        free(root);
    }
}

int main() {
    int n = 50000;
    TrieNode* root = createNode();
    
    srand(42);
    clock_t start = clock();
    
    for (int i = 0; i < n; i++) {
        char word[10];
        int len = rand() % 8 + 2;
        for (int j = 0; j < len; j++)
            word[j] = 'a' + (rand() % 26);
        word[len] = '\0';
        insert(root, word);
    }
    
    srand(42);
    int found = 0;
    for (int i = 0; i < n; i++) {
        char word[10];
        int len = rand() % 8 + 2;
        for (int j = 0; j < len; j++)
            word[j] = 'a' + (rand() % 26);
        word[len] = '\0';
        if (search(root, word))
            found++;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Trie: %d inserts+searches in %.6f seconds (%d found)\n", 
           n, time_spent, found);
    
    freeTrie(root);
    return 0;
}
