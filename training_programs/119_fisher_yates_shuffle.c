// Fisher-Yates shuffle algorithm (Knuth shuffle)
// Unbiased random permutation generation in O(n) time
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE 100000
#define NUM_SHUFFLES 500

void fisher_yates_shuffle(int *array, int n, unsigned int *seed) {
    for (int i = n - 1; i > 0; i--) {
        *seed = *seed * 1103515245 + 12345;
        int j = (*seed) % (i + 1);
        
        // Swap array[i] and array[j]
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

// Inside-out variant (can generate permutation on-the-fly)
void fisher_yates_inside_out(int *array, int n, unsigned int *seed) {
    array[0] = 0;
    for (int i = 1; i < n; i++) {
        *seed = *seed * 1103515245 + 12345;
        int j = (*seed) % (i + 1);
        
        if (j != i) {
            array[i] = array[j];
        }
        array[j] = i;
    }
}

void init_array(int *array, int n) {
    for (int i = 0; i < n; i++) {
        array[i] = i;
    }
}

int verify_permutation(int *array, int n) {
    int *seen = (int*)calloc(n, sizeof(int));
    
    for (int i = 0; i < n; i++) {
        if (array[i] < 0 || array[i] >= n || seen[array[i]]) {
            free(seen);
            return 0;
        }
        seen[array[i]] = 1;
    }
    
    free(seen);
    return 1;
}

int main() {
    int *array = (int*)malloc(ARRAY_SIZE * sizeof(int));
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    int valid_shuffles = 0;
    for (int trial = 0; trial < NUM_SHUFFLES; trial++) {
        init_array(array, ARRAY_SIZE);
        
        if (trial % 2 == 0) {
            fisher_yates_shuffle(array, ARRAY_SIZE, &seed);
        } else {
            init_array(array, ARRAY_SIZE);
            fisher_yates_inside_out(array, ARRAY_SIZE, &seed);
        }
        
        if (verify_permutation(array, ARRAY_SIZE)) {
            valid_shuffles++;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Calculate some statistics on final shuffle
    long long sum = 0;
    for (int i = 0; i < 100; i++) {
        sum += array[i];
    }
    
    printf("Fisher-Yates shuffle: array=%d, %d shuffles, %.6f seconds\n",
           ARRAY_SIZE, NUM_SHUFFLES, time_spent);
    printf("Valid permutations: %d/%d\n", valid_shuffles, NUM_SHUFFLES);
    printf("First 100 elements sum: %lld\n", sum);
    
    free(array);
    
    return 0;
}
