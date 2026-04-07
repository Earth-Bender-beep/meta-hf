// Burrows-Wheeler Transform (BWT) for data compression
// Used in bzip2 and other compression algorithms
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TEXT_SIZE 10000
#define NUM_TRANSFORMS 200

typedef struct {
    char *rotation;
    int index;
} Rotation;

int compare_rotations(const void *a, const void *b) {
    Rotation *ra = (Rotation*)a;
    Rotation *rb = (Rotation*)b;
    return strcmp(ra->rotation, rb->rotation);
}

void burrows_wheeler_transform(const char *input, char *output, int *primary_index) {
    int len = strlen(input);
    
    // Create all rotations
    Rotation *rotations = (Rotation*)malloc(len * sizeof(Rotation));
    for (int i = 0; i < len; i++) {
        rotations[i].rotation = (char*)malloc(len + 1);
        rotations[i].index = i;
        
        // Create rotation
        for (int j = 0; j < len; j++) {
            rotations[i].rotation[j] = input[(i + j) % len];
        }
        rotations[i].rotation[len] = '\0';
    }
    
    // Sort rotations
    qsort(rotations, len, sizeof(Rotation), compare_rotations);
    
    // Extract last column and find primary index
    for (int i = 0; i < len; i++) {
        output[i] = rotations[i].rotation[len - 1];
        if (rotations[i].index == 0) {
            *primary_index = i;
        }
    }
    output[len] = '\0';
    
    // Free memory
    for (int i = 0; i < len; i++) {
        free(rotations[i].rotation);
    }
    free(rotations);
}

void inverse_burrows_wheeler_transform(const char *input, char *output, int primary_index) {
    int len = strlen(input);
    
    // Count occurrences of each character
    int count[256] = {0};
    for (int i = 0; i < len; i++) {
        count[(unsigned char)input[i]]++;
    }
    
    // Compute cumulative counts
    int cumulative[256] = {0};
    int total = 0;
    for (int i = 0; i < 256; i++) {
        cumulative[i] = total;
        total += count[i];
    }
    
    // Build transformation vector
    int *transform = (int*)malloc(len * sizeof(int));
    for (int i = 0; i < len; i++) {
        unsigned char c = input[i];
        transform[cumulative[c]++] = i;
    }
    
    // Reconstruct original string
    int idx = primary_index;
    for (int i = 0; i < len; i++) {
        output[i] = input[idx];
        idx = transform[idx];
    }
    output[len] = '\0';
    
    free(transform);
}

// Move-to-front encoding (often used with BWT)
void move_to_front_encode(const char *input, unsigned char *output) {
    int len = strlen(input);
    unsigned char table[256];
    
    // Initialize table
    for (int i = 0; i < 256; i++) {
        table[i] = i;
    }
    
    for (int i = 0; i < len; i++) {
        unsigned char c = (unsigned char)input[i];
        
        // Find position of character
        int pos = 0;
        while (table[pos] != c) pos++;
        
        output[i] = pos;
        
        // Move to front
        for (int j = pos; j > 0; j--) {
            table[j] = table[j - 1];
        }
        table[0] = c;
    }
}

void move_to_front_decode(const unsigned char *input, char *output, int len) {
    unsigned char table[256];
    
    // Initialize table
    for (int i = 0; i < 256; i++) {
        table[i] = i;
    }
    
    for (int i = 0; i < len; i++) {
        int pos = input[i];
        unsigned char c = table[pos];
        output[i] = c;
        
        // Move to front
        for (int j = pos; j > 0; j--) {
            table[j] = table[j - 1];
        }
        table[0] = c;
    }
    output[len] = '\0';
}

void generate_text(char *text, int size) {
    unsigned int seed = 42;
    for (int i = 0; i < size - 1; i++) {
        seed = seed * 1103515245 + 12345;
        // Use biased distribution for better compression
        int r = seed % 100;
        if (r < 40) {
            text[i] = 'a';
        } else if (r < 70) {
            text[i] = 'b';
        } else if (r < 85) {
            text[i] = 'c';
        } else {
            text[i] = 'd' + (seed % 4);
        }
    }
    text[size - 1] = '\0';
}

int main() {
    char *input = (char*)malloc(TEXT_SIZE);
    char *bwt_output = (char*)malloc(TEXT_SIZE);
    char *reconstructed = (char*)malloc(TEXT_SIZE);
    unsigned char *mtf_output = (unsigned char*)malloc(TEXT_SIZE);
    
    generate_text(input, TEXT_SIZE);
    
    clock_t start = clock();
    
    int successful = 0;
    
    for (int iter = 0; iter < NUM_TRANSFORMS; iter++) {
        int primary_index;
        
        // Forward BWT
        burrows_wheeler_transform(input, bwt_output, &primary_index);
        
        // MTF encoding
        move_to_front_encode(bwt_output, mtf_output);
        
        // Inverse BWT
        inverse_burrows_wheeler_transform(bwt_output, reconstructed, primary_index);
        
        // Verify
        if (strcmp(input, reconstructed) == 0) {
            successful++;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Burrows-Wheeler Transform: text=%d, %d transforms, %.6f seconds\n",
           TEXT_SIZE, NUM_TRANSFORMS, time_spent);
    printf("Successful reconstructions: %d/%d\n", successful, NUM_TRANSFORMS);
    
    free(input);
    free(bwt_output);
    free(reconstructed);
    free(mtf_output);
    
    return 0;
}
