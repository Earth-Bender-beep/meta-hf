// LRU (Least Recently Used) Cache implementation
// Using hash table and doubly linked list for O(1) operations
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CACHE_CAPACITY 1000
#define HASH_SIZE 2048
#define NUM_OPERATIONS 100000

typedef struct Node {
    int key;
    int value;
    struct Node *prev;
    struct Node *next;
} Node;

typedef struct {
    Node **hash_table;
    Node *head;
    Node *tail;
    int capacity;
    int size;
} LRUCache;

Node* create_node(int key, int value) {
    Node *node = (Node*)malloc(sizeof(Node));
    node->key = key;
    node->value = value;
    node->prev = NULL;
    node->next = NULL;
    return node;
}

LRUCache* create_cache(int capacity) {
    LRUCache *cache = (LRUCache*)malloc(sizeof(LRUCache));
    cache->hash_table = (Node**)calloc(HASH_SIZE, sizeof(Node*));
    cache->head = create_node(0, 0);  // Dummy head
    cache->tail = create_node(0, 0);  // Dummy tail
    cache->head->next = cache->tail;
    cache->tail->prev = cache->head;
    cache->capacity = capacity;
    cache->size = 0;
    return cache;
}

void move_to_front(LRUCache *cache, Node *node) {
    // Remove from current position
    node->prev->next = node->next;
    node->next->prev = node->prev;
    
    // Insert after head
    node->next = cache->head->next;
    node->prev = cache->head;
    cache->head->next->prev = node;
    cache->head->next = node;
}

int get(LRUCache *cache, int key) {
    int hash = key % HASH_SIZE;
    Node *node = cache->hash_table[hash];
    
    // Search in hash bucket
    while (node) {
        if (node->key == key) {
            move_to_front(cache, node);
            return node->value;
        }
        node = node->next;
    }
    
    return -1;  // Not found
}

void put(LRUCache *cache, int key, int value) {
    int hash = key % HASH_SIZE;
    
    // Check if key exists
    Node *node = cache->hash_table[hash];
    Node *prev = NULL;
    
    while (node) {
        if (node->key == key) {
            node->value = value;
            move_to_front(cache, node);
            return;
        }
        prev = node;
        node = node->next;
    }
    
    // Key doesn't exist - create new node
    Node *new_node = create_node(key, value);
    
    // Add to hash table
    if (!cache->hash_table[hash]) {
        cache->hash_table[hash] = new_node;
    } else {
        prev->next = new_node;
    }
    
    // Add to front of list
    new_node->next = cache->head->next;
    new_node->prev = cache->head;
    cache->head->next->prev = new_node;
    cache->head->next = new_node;
    
    cache->size++;
    
    // Evict if necessary
    if (cache->size > cache->capacity) {
        Node *lru = cache->tail->prev;
        
        // Remove from list
        lru->prev->next = cache->tail;
        cache->tail->prev = lru->prev;
        
        // Remove from hash table
        int lru_hash = lru->key % HASH_SIZE;
        Node *h = cache->hash_table[lru_hash];
        Node *h_prev = NULL;
        
        while (h) {
            if (h->key == lru->key) {
                if (h_prev) {
                    h_prev->next = h->next;
                } else {
                    cache->hash_table[lru_hash] = h->next;
                }
                break;
            }
            h_prev = h;
            h = h->next;
        }
        
        free(lru);
        cache->size--;
    }
}

int main() {
    LRUCache *cache = create_cache(CACHE_CAPACITY);
    
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    int hits = 0, misses = 0;
    
    for (int i = 0; i < NUM_OPERATIONS; i++) {
        seed = seed * 1103515245 + 12345;
        int operation = seed % 100;
        seed = seed * 1103515245 + 12345;
        int key = seed % (CACHE_CAPACITY * 2);
        
        if (operation < 70) {  // 70% puts
            seed = seed * 1103515245 + 12345;
            int value = seed % 10000;
            put(cache, key, value);
        } else {  // 30% gets
            int result = get(cache, key);
            if (result != -1) {
                hits++;
            } else {
                misses++;
            }
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("LRU Cache: capacity=%d, %d operations, %.6f seconds\n",
           CACHE_CAPACITY, NUM_OPERATIONS, time_spent);
    printf("Cache hits: %d, misses: %d, hit rate: %.2f%%\n",
           hits, misses, 100.0 * hits / (hits + misses));
    
    return 0;
}
