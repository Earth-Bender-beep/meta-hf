// Reservoir sampling algorithm
// Randomly select k items from stream of unknown size with uniform probability
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STREAM_SIZE 1000000
#define RESERVOIR_SIZE 1000
#define NUM_TRIALS 100

void reservoir_sample(int *stream, int stream_len, int *reservoir, int k) {
    // Initialize reservoir with first k elements
    for (int i = 0; i < k && i < stream_len; i++) {
        reservoir[i] = stream[i];
    }
    
    unsigned int seed = 42;
    
    // Process remaining elements
    for (int i = k; i < stream_len; i++) {
        // Generate random index
        seed = seed * 1103515245 + 12345;
        int j = seed % (i + 1);
        
        // Replace element with decreasing probability
        if (j < k) {
            reservoir[j] = stream[i];
        }
    }
}

void generate_stream(int *stream, int size) {
    for (int i = 0; i < size; i++) {
        stream[i] = i + 1;
    }
}

int main() {
    int *stream = (int*)malloc(STREAM_SIZE * sizeof(int));
    int *reservoir = (int*)malloc(RESERVOIR_SIZE * sizeof(int));
    
    generate_stream(stream, STREAM_SIZE);
    
    clock_t start = clock();
    
    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        reservoir_sample(stream, STREAM_SIZE, reservoir, RESERVOIR_SIZE);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Calculate statistics on final sample
    long long sum = 0;
    for (int i = 0; i < RESERVOIR_SIZE; i++) {
        sum += reservoir[i];
    }
    
    printf("Reservoir sampling: stream=%d, k=%d, %d trials, %.6f seconds\n",
           STREAM_SIZE, RESERVOIR_SIZE, NUM_TRIALS, time_spent);
    printf("Final sample average: %.2f\n", sum / (double)RESERVOIR_SIZE);
    
    free(stream);
    free(reservoir);
    
    return 0;
}
