// Rolling hash for efficient string matching (Rabin fingerprint)
// Used in rsync, plagiarism detection, and data deduplication
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TEXT_SIZE 100000
#define WINDOW_SIZE 100
#define NUM_SEARCHES 10000
#define BASE 256
#define MOD 1000000007

typedef struct {
    unsigned long long hash;
    int window_size;
    unsigned long long base_power;
} RollingHash;

void init_rolling_hash(RollingHash *rh, int window_size) {
    rh->hash = 0;
    rh->window_size = window_size;
    rh->base_power = 1;
    
    // Compute BASE^(window_size-1) % MOD
    for (int i = 0; i < window_size - 1; i++) {
        rh->base_power = (rh->base_power * BASE) % MOD;
    }
}

unsigned long long compute_hash(const char *str, int len) {
    unsigned long long hash = 0;
    for (int i = 0; i < len; i++) {
        hash = (hash * BASE + (unsigned char)str[i]) % MOD;
    }
    return hash;
}

void roll_hash(RollingHash *rh, char old_char, char new_char) {
    // Remove old character
    rh->hash = (rh->hash + MOD - (rh->base_power * (unsigned char)old_char) % MOD) % MOD;
    
    // Shift and add new character
    rh->hash = (rh->hash * BASE + (unsigned char)new_char) % MOD;
}

int rolling_hash_search(const char *text, int text_len, const char *pattern, int pattern_len) {
    if (pattern_len > text_len) return 0;
    
    RollingHash rh;
    init_rolling_hash(&rh, pattern_len);
    
    unsigned long long pattern_hash = compute_hash(pattern, pattern_len);
    rh.hash = compute_hash(text, pattern_len);
    
    int matches = 0;
    
    // Check first window
    if (rh.hash == pattern_hash) {
        if (strncmp(text, pattern, pattern_len) == 0) {
            matches++;
        }
    }
    
    // Roll through the text
    for (int i = pattern_len; i < text_len; i++) {
        roll_hash(&rh, text[i - pattern_len], text[i]);
        
        if (rh.hash == pattern_hash) {
            if (strncmp(text + i - pattern_len + 1, pattern, pattern_len) == 0) {
                matches++;
            }
        }
    }
    
    return matches;
}

// Chunking for data deduplication
typedef struct {
    unsigned long long hash;
    int offset;
    int length;
} Chunk;

int chunk_data(const char *data, int data_len, int window_size, Chunk *chunks, int max_chunks) {
    RollingHash rh;
    init_rolling_hash(&rh, window_size);
    
    rh.hash = compute_hash(data, window_size);
    
    int chunk_count = 0;
    int last_boundary = 0;
    
    for (int i = window_size; i < data_len; i++) {
        roll_hash(&rh, data[i - window_size], data[i]);
        
        // Use low bits to determine chunk boundaries
        if ((rh.hash & 0xFF) == 0 && chunk_count < max_chunks) {
            chunks[chunk_count].hash = compute_hash(data + last_boundary, i - last_boundary);
            chunks[chunk_count].offset = last_boundary;
            chunks[chunk_count].length = i - last_boundary;
            chunk_count++;
            last_boundary = i;
        }
    }
    
    // Add final chunk
    if (last_boundary < data_len && chunk_count < max_chunks) {
        chunks[chunk_count].hash = compute_hash(data + last_boundary, data_len - last_boundary);
        chunks[chunk_count].offset = last_boundary;
        chunks[chunk_count].length = data_len - last_boundary;
        chunk_count++;
    }
    
    return chunk_count;
}

void generate_text(char *text, int size) {
    unsigned int seed = 42;
    for (int i = 0; i < size - 1; i++) {
        seed = seed * 1103515245 + 12345;
        text[i] = 'a' + (seed % 26);
    }
    text[size - 1] = '\0';
}

int main() {
    char *text = (char*)malloc(TEXT_SIZE);
    char *pattern = (char*)malloc(WINDOW_SIZE + 1);
    
    generate_text(text, TEXT_SIZE);
    
    clock_t start = clock();
    
    int total_matches = 0;
    
    // Pattern searching
    for (int i = 0; i < NUM_SEARCHES; i++) {
        int pos = (i * 137) % (TEXT_SIZE - WINDOW_SIZE);
        strncpy(pattern, text + pos, WINDOW_SIZE);
        pattern[WINDOW_SIZE] = '\0';
        
        total_matches += rolling_hash_search(text, TEXT_SIZE, pattern, WINDOW_SIZE);
    }
    
    // Data chunking for deduplication
    Chunk *chunks = (Chunk*)malloc(1000 * sizeof(Chunk));
    int chunk_count = chunk_data(text, TEXT_SIZE, 32, chunks, 1000);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Rolling hash: text=%d, window=%d, %d searches, %.6f seconds\n",
           TEXT_SIZE, WINDOW_SIZE, NUM_SEARCHES, time_spent);
    printf("Total matches: %d, Chunks: %d\n", total_matches, chunk_count);
    
    free(text);
    free(pattern);
    free(chunks);
    
    return 0;
}
