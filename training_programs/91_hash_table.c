// Hash table with chaining for collision resolution
// Dynamic hashing, linked lists, load factor management
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TABLE_SIZE 1000
#define NUM_OPERATIONS 10000

typedef struct Entry {
    int key;
    int value;
    struct Entry *next;
} Entry;

typedef struct {
    Entry **buckets;
    int size;
    int count;
} HashTable;

unsigned int hash_function(int key, int table_size) {
    return (unsigned int)key % table_size;
}

HashTable* create_hash_table(int size) {
    HashTable *ht = (HashTable*)malloc(sizeof(HashTable));
    ht->size = size;
    ht->count = 0;
    ht->buckets = (Entry**)calloc(size, sizeof(Entry*));
    return ht;
}

void hash_insert(HashTable *ht, int key, int value) {
    unsigned int index = hash_function(key, ht->size);
    
    Entry *current = ht->buckets[index];
    while (current) {
        if (current->key == key) {
            current->value = value;
            return;
        }
        current = current->next;
    }
    
    Entry *new_entry = (Entry*)malloc(sizeof(Entry));
    new_entry->key = key;
    new_entry->value = value;
    new_entry->next = ht->buckets[index];
    ht->buckets[index] = new_entry;
    ht->count++;
}

int hash_search(HashTable *ht, int key, int *value) {
    unsigned int index = hash_function(key, ht->size);
    
    Entry *current = ht->buckets[index];
    while (current) {
        if (current->key == key) {
            *value = current->value;
            return 1;
        }
        current = current->next;
    }
    
    return 0;
}

void hash_delete(HashTable *ht, int key) {
    unsigned int index = hash_function(key, ht->size);
    
    Entry *current = ht->buckets[index];
    Entry *prev = NULL;
    
    while (current) {
        if (current->key == key) {
            if (prev) {
                prev->next = current->next;
            } else {
                ht->buckets[index] = current->next;
            }
            free(current);
            ht->count--;
            return;
        }
        prev = current;
        current = current->next;
    }
}

void free_hash_table(HashTable *ht) {
    for (int i = 0; i < ht->size; i++) {
        Entry *current = ht->buckets[i];
        while (current) {
            Entry *temp = current;
            current = current->next;
            free(temp);
        }
    }
    free(ht->buckets);
    free(ht);
}

int main() {
    HashTable *ht = create_hash_table(TABLE_SIZE);
    
    clock_t start = clock();
    
    // Insert operations
    for (int i = 0; i < NUM_OPERATIONS; i++) {
        hash_insert(ht, (i * 17) % 50000, i);
    }
    
    // Search operations
    int found = 0;
    int value;
    for (int i = 0; i < NUM_OPERATIONS; i++) {
        if (hash_search(ht, (i * 17) % 50000, &value)) {
            found++;
        }
    }
    
    // Delete operations
    for (int i = 0; i < NUM_OPERATIONS / 2; i++) {
        hash_delete(ht, (i * 17) % 50000);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Hash table: %d operations, %.6f seconds\n", NUM_OPERATIONS, time_spent);
    printf("Found: %d, Final count: %d\n", found, ht->count);
    
    free_hash_table(ht);
    return 0;
}
