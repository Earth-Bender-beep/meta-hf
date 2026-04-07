// Trie with autocomplete functionality
// Efficient prefix-based word search and suggestion
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ALPHABET_SIZE 26
#define MAX_WORD_LEN 50
#define NUM_WORDS 1000
#define NUM_QUERIES 5000

typedef struct TrieNode {
    struct TrieNode *children[ALPHABET_SIZE];
    int is_end_of_word;
    int word_count;  // Number of words with this prefix
} TrieNode;

TrieNode* create_node() {
    TrieNode *node = (TrieNode*)malloc(sizeof(TrieNode));
    node->is_end_of_word = 0;
    node->word_count = 0;
    for (int i = 0; i < ALPHABET_SIZE; i++) {
        node->children[i] = NULL;
    }
    return node;
}

void insert_word(TrieNode *root, const char *word) {
    TrieNode *curr = root;
    
    for (int i = 0; word[i] != '\0'; i++) {
        int index = word[i] - 'a';
        if (index < 0 || index >= ALPHABET_SIZE) continue;
        
        if (!curr->children[index]) {
            curr->children[index] = create_node();
        }
        
        curr = curr->children[index];
        curr->word_count++;
    }
    
    curr->is_end_of_word = 1;
}

int search_word(TrieNode *root, const char *word) {
    TrieNode *curr = root;
    
    for (int i = 0; word[i] != '\0'; i++) {
        int index = word[i] - 'a';
        if (index < 0 || index >= ALPHABET_SIZE || !curr->children[index]) {
            return 0;
        }
        curr = curr->children[index];
    }
    
    return curr->is_end_of_word;
}

int count_words_with_prefix(TrieNode *root, const char *prefix) {
    TrieNode *curr = root;
    
    for (int i = 0; prefix[i] != '\0'; i++) {
        int index = prefix[i] - 'a';
        if (index < 0 || index >= ALPHABET_SIZE || !curr->children[index]) {
            return 0;
        }
        curr = curr->children[index];
    }
    
    return curr->word_count;
}

void collect_words(TrieNode *node, char *prefix, int depth, char results[][MAX_WORD_LEN], int *count, int max_results) {
    if (*count >= max_results) return;
    
    if (node->is_end_of_word) {
        prefix[depth] = '\0';
        strcpy(results[*count], prefix);
        (*count)++;
    }
    
    for (int i = 0; i < ALPHABET_SIZE; i++) {
        if (node->children[i]) {
            prefix[depth] = 'a' + i;
            collect_words(node->children[i], prefix, depth + 1, results, count, max_results);
        }
    }
}

int autocomplete(TrieNode *root, const char *prefix, char results[][MAX_WORD_LEN], int max_results) {
    TrieNode *curr = root;
    
    // Navigate to prefix node
    for (int i = 0; prefix[i] != '\0'; i++) {
        int index = prefix[i] - 'a';
        if (index < 0 || index >= ALPHABET_SIZE || !curr->children[index]) {
            return 0;
        }
        curr = curr->children[index];
    }
    
    // Collect words
    char word_buffer[MAX_WORD_LEN];
    strcpy(word_buffer, prefix);
    int count = 0;
    collect_words(curr, word_buffer, strlen(prefix), results, &count, max_results);
    
    return count;
}

void free_trie(TrieNode *node) {
    if (!node) return;
    
    for (int i = 0; i < ALPHABET_SIZE; i++) {
        if (node->children[i]) {
            free_trie(node->children[i]);
        }
    }
    
    free(node);
}

void generate_word(char *word, int len) {
    static unsigned int seed = 42;
    for (int i = 0; i < len - 1; i++) {
        seed = seed * 1103515245 + 12345;
        word[i] = 'a' + (seed % 26);
    }
    word[len - 1] = '\0';
}

int main() {
    TrieNode *root = create_node();
    
    // Generate and insert words
    char **words = (char**)malloc(NUM_WORDS * sizeof(char*));
    for (int i = 0; i < NUM_WORDS; i++) {
        words[i] = (char*)malloc(MAX_WORD_LEN);
        int len = 3 + (i % 10);
        generate_word(words[i], len);
        insert_word(root, words[i]);
    }
    
    clock_t start = clock();
    
    int total_suggestions = 0;
    char results[10][MAX_WORD_LEN];
    
    for (int q = 0; q < NUM_QUERIES; q++) {
        // Random prefix query
        int word_idx = q % NUM_WORDS;
        int prefix_len = 1 + (q % 4);
        
        char prefix[MAX_WORD_LEN];
        strncpy(prefix, words[word_idx], prefix_len);
        prefix[prefix_len] = '\0';
        
        int count = autocomplete(root, prefix, results, 10);
        total_suggestions += count;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Trie autocomplete: %d words, %d queries, %.6f seconds\n",
           NUM_WORDS, NUM_QUERIES, time_spent);
    printf("Average suggestions per query: %.2f\n", total_suggestions / (double)NUM_QUERIES);
    
    for (int i = 0; i < NUM_WORDS; i++) {
        free(words[i]);
    }
    free(words);
    free_trie(root);
    
    return 0;
}
