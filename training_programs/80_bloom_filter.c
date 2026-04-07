// Bloom filter - probabilistic membership test
// Multiple hash functions, bit array, space-efficient
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BLOOM_SIZE 10000
#define NUM_HASHES 5
#define NUM_INSERTS 2000
#define NUM_QUERIES 5000

typedef struct {
    unsigned char *bits;
    int size;
    int num_hashes;
} BloomFilter;

BloomFilter* create_bloom_filter(int size, int num_hashes) {
    BloomFilter *bf = (BloomFilter*)malloc(sizeof(BloomFilter));
    bf->size = size;
    bf->num_hashes = num_hashes;
    bf->bits = (unsigned char*)calloc((size + 7) / 8, sizeof(unsigned char));
    return bf;
}

void set_bit(BloomFilter *bf, int pos) {
    bf->bits[pos / 8] |= (1 << (pos % 8));
}

int get_bit(BloomFilter *bf, int pos) {
    return (bf->bits[pos / 8] & (1 << (pos % 8))) != 0;
}

unsigned int hash_function(const char *str, int seed) {
    unsigned int hash = seed;
    while (*str) {
        hash = hash * 31 + (*str);
        str++;
    }
    return hash;
}

void bloom_insert(BloomFilter *bf, const char *item) {
    for (int i = 0; i < bf->num_hashes; i++) {
        unsigned int hash = hash_function(item, i * 7919);
        int pos = hash % bf->size;
        set_bit(bf, pos);
    }
}

int bloom_query(BloomFilter *bf, const char *item) {
    for (int i = 0; i < bf->num_hashes; i++) {
        unsigned int hash = hash_function(item, i * 7919);
        int pos = hash % bf->size;
        if (!get_bit(bf, pos)) {
            return 0;
        }
    }
    return 1;
}

void free_bloom_filter(BloomFilter *bf) {
    free(bf->bits);
    free(bf);
}

void generate_string(char *str, int len, int seed) {
    for (int i = 0; i < len - 1; i++) {
        str[i] = 'a' + ((seed + i * 7) % 26);
    }
    str[len - 1] = '\0';
}

int main() {
    BloomFilter *bf = create_bloom_filter(BLOOM_SIZE, NUM_HASHES);
    
    char **inserted = (char**)malloc(NUM_INSERTS * sizeof(char*));
    char **queries = (char**)malloc(NUM_QUERIES * sizeof(char*));
    
    for (int i = 0; i < NUM_INSERTS; i++) {
        inserted[i] = (char*)malloc(20);
        generate_string(inserted[i], 20, i);
    }
    
    for (int i = 0; i < NUM_QUERIES; i++) {
        queries[i] = (char*)malloc(20);
        generate_string(queries[i], 20, i + 10000);
    }
    
    clock_t start = clock();
    
    // Insert items
    for (int i = 0; i < NUM_INSERTS; i++) {
        bloom_insert(bf, inserted[i]);
    }
    
    // Query items
    int true_positives = 0;
    int false_positives = 0;
    
    for (int i = 0; i < NUM_INSERTS; i++) {
        if (bloom_query(bf, inserted[i])) {
            true_positives++;
        }
    }
    
    for (int i = 0; i < NUM_QUERIES; i++) {
        if (bloom_query(bf, queries[i])) {
            false_positives++;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Bloom filter: size=%d, hashes=%d, %.6f seconds\n",
           BLOOM_SIZE, NUM_HASHES, time_spent);
    printf("True positives: %d/%d, False positives: %d/%d (%.2f%%)\n",
           true_positives, NUM_INSERTS, false_positives, NUM_QUERIES,
           100.0 * false_positives / NUM_QUERIES);
    
    for (int i = 0; i < NUM_INSERTS; i++) free(inserted[i]);
    for (int i = 0; i < NUM_QUERIES; i++) free(queries[i]);
    free(inserted);
    free(queries);
    free_bloom_filter(bf);
    
    return 0;
}
