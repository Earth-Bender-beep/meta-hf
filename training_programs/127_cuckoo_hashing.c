// Cuckoo hashing - uses two hash functions and two tables
// Provides O(1) worst-case lookup with high probability
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TABLE_SIZE 5000
#define MAX_REHASH 500
#define NUM_OPERATIONS 20000

typedef struct {
    int key;
    int value;
    int occupied;
} Entry;

typedef struct {
    Entry *table1;
    Entry *table2;
    int size;
} CuckooHash;

unsigned int hash_seed1 = 42;
unsigned int hash_seed2 = 123;

int hash1(int key, int size) {
    return ((key * 2654435761u) ^ hash_seed1) % size;
}

int hash2(int key, int size) {
    return ((key * 2246822519u) ^ hash_seed2) % size;
}

CuckooHash* create_cuckoo_hash(int size) {
    CuckooHash *ch = (CuckooHash*)malloc(sizeof(CuckooHash));
    ch->table1 = (Entry*)calloc(size, sizeof(Entry));
    ch->table2 = (Entry*)calloc(size, sizeof(Entry));
    ch->size = size;
    return ch;
}

int cuckoo_search(CuckooHash *ch, int key) {
    int h1 = hash1(key, ch->size);
    if (ch->table1[h1].occupied && ch->table1[h1].key == key) {
        return ch->table1[h1].value;
    }
    
    int h2 = hash2(key, ch->size);
    if (ch->table2[h2].occupied && ch->table2[h2].key == key) {
        return ch->table2[h2].value;
    }
    
    return -1;  // Not found
}

int cuckoo_insert(CuckooHash *ch, int key, int value) {
    // Check if key already exists
    int h1 = hash1(key, ch->size);
    if (ch->table1[h1].occupied && ch->table1[h1].key == key) {
        ch->table1[h1].value = value;
        return 1;
    }
    
    int h2 = hash2(key, ch->size);
    if (ch->table2[h2].occupied && ch->table2[h2].key == key) {
        ch->table2[h2].value = value;
        return 1;
    }
    
    // Insert new key
    int curr_key = key;
    int curr_value = value;
    int use_table1 = 1;
    
    for (int i = 0; i < MAX_REHASH; i++) {
        if (use_table1) {
            h1 = hash1(curr_key, ch->size);
            
            if (!ch->table1[h1].occupied) {
                ch->table1[h1].key = curr_key;
                ch->table1[h1].value = curr_value;
                ch->table1[h1].occupied = 1;
                return 1;
            }
            
            // Evict and swap
            int temp_key = ch->table1[h1].key;
            int temp_value = ch->table1[h1].value;
            ch->table1[h1].key = curr_key;
            ch->table1[h1].value = curr_value;
            curr_key = temp_key;
            curr_value = temp_value;
        } else {
            h2 = hash2(curr_key, ch->size);
            
            if (!ch->table2[h2].occupied) {
                ch->table2[h2].key = curr_key;
                ch->table2[h2].value = curr_value;
                ch->table2[h2].occupied = 1;
                return 1;
            }
            
            // Evict and swap
            int temp_key = ch->table2[h2].key;
            int temp_value = ch->table2[h2].value;
            ch->table2[h2].key = curr_key;
            ch->table2[h2].value = curr_value;
            curr_key = temp_key;
            curr_value = temp_value;
        }
        
        use_table1 = !use_table1;
    }
    
    return 0;  // Failed to insert (needs rehashing)
}

int main() {
    CuckooHash *ch = create_cuckoo_hash(TABLE_SIZE);
    
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    int insertions = 0;
    int searches = 0;
    int found = 0;
    
    for (int i = 0; i < NUM_OPERATIONS; i++) {
        seed = seed * 1103515245 + 12345;
        int op = seed % 100;
        seed = seed * 1103515245 + 12345;
        int key = seed % (TABLE_SIZE * 2);
        
        if (op < 60) {  // 60% insertions
            seed = seed * 1103515245 + 12345;
            int value = seed % 10000;
            if (cuckoo_insert(ch, key, value)) {
                insertions++;
            }
        } else {  // 40% searches
            searches++;
            if (cuckoo_search(ch, key) != -1) {
                found++;
            }
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Cuckoo hashing: table_size=%d, %d operations, %.6f seconds\n",
           TABLE_SIZE, NUM_OPERATIONS, time_spent);
    printf("Insertions: %d, Searches: %d, Found: %d\n", insertions, searches, found);
    
    free(ch->table1);
    free(ch->table2);
    free(ch);
    
    return 0;
}
