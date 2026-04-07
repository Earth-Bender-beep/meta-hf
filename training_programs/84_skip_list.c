// Skip list - probabilistic data structure for sorted data
// Multiple levels, randomized structure, O(log n) expected time
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_LEVEL 16
#define P_FACTOR 0.5
#define N_OPERATIONS 3000

typedef struct SkipNode {
    int key;
    struct SkipNode **forward;
} SkipNode;

typedef struct {
    int level;
    SkipNode *header;
} SkipList;

int random_level() {
    int level = 1;
    while ((rand() / (double)RAND_MAX) < P_FACTOR && level < MAX_LEVEL) {
        level++;
    }
    return level;
}

SkipNode* create_node(int key, int level) {
    SkipNode *node = (SkipNode*)malloc(sizeof(SkipNode));
    node->key = key;
    node->forward = (SkipNode**)malloc((level + 1) * sizeof(SkipNode*));
    for (int i = 0; i <= level; i++) {
        node->forward[i] = NULL;
    }
    return node;
}

SkipList* create_skip_list() {
    SkipList *list = (SkipList*)malloc(sizeof(SkipList));
    list->level = 0;
    list->header = create_node(-1, MAX_LEVEL);
    return list;
}

void skip_insert(SkipList *list, int key) {
    SkipNode *update[MAX_LEVEL + 1];
    SkipNode *current = list->header;
    
    for (int i = list->level; i >= 0; i--) {
        while (current->forward[i] && current->forward[i]->key < key) {
            current = current->forward[i];
        }
        update[i] = current;
    }
    
    current = current->forward[0];
    
    if (!current || current->key != key) {
        int new_level = random_level();
        
        if (new_level > list->level) {
            for (int i = list->level + 1; i <= new_level; i++) {
                update[i] = list->header;
            }
            list->level = new_level;
        }
        
        SkipNode *new_node = create_node(key, new_level);
        
        for (int i = 0; i <= new_level; i++) {
            new_node->forward[i] = update[i]->forward[i];
            update[i]->forward[i] = new_node;
        }
    }
}

int skip_search(SkipList *list, int key) {
    SkipNode *current = list->header;
    
    for (int i = list->level; i >= 0; i--) {
        while (current->forward[i] && current->forward[i]->key < key) {
            current = current->forward[i];
        }
    }
    
    current = current->forward[0];
    
    return current && current->key == key;
}

void skip_delete(SkipList *list, int key) {
    SkipNode *update[MAX_LEVEL + 1];
    SkipNode *current = list->header;
    
    for (int i = list->level; i >= 0; i--) {
        while (current->forward[i] && current->forward[i]->key < key) {
            current = current->forward[i];
        }
        update[i] = current;
    }
    
    current = current->forward[0];
    
    if (current && current->key == key) {
        for (int i = 0; i <= list->level; i++) {
            if (update[i]->forward[i] != current) break;
            update[i]->forward[i] = current->forward[i];
        }
        
        free(current->forward);
        free(current);
        
        while (list->level > 0 && !list->header->forward[list->level]) {
            list->level--;
        }
    }
}

void free_skip_list(SkipList *list) {
    SkipNode *current = list->header->forward[0];
    
    while (current) {
        SkipNode *temp = current;
        current = current->forward[0];
        free(temp->forward);
        free(temp);
    }
    
    free(list->header->forward);
    free(list->header);
    free(list);
}

int main() {
    srand(42);
    SkipList *list = create_skip_list();
    
    clock_t start = clock();
    
    // Insert operations
    for (int i = 0; i < N_OPERATIONS; i++) {
        skip_insert(list, (i * 13 + 7) % 5000);
    }
    
    // Search operations
    int found = 0;
    for (int i = 0; i < N_OPERATIONS; i++) {
        if (skip_search(list, i % 5000)) found++;
    }
    
    // Delete operations
    for (int i = 0; i < N_OPERATIONS / 2; i++) {
        skip_delete(list, (i * 5) % 5000);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Skip list: %d operations, %.6f seconds\n", N_OPERATIONS, time_spent);
    printf("Items found: %d, Max level: %d\n", found, list->level);
    
    free_skip_list(list);
    return 0;
}
