// Parallel merge sort simulation (single-threaded but shows merge pattern)
// Demonstrates divide-and-conquer with multiple independent merge phases
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ARRAY_SIZE 100000
#define NUM_TRIALS 20

void merge(int arr[], int left, int mid, int right, int temp[]) {
    int i = left, j = mid + 1, k = left;
    
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    
    while (i <= mid) {
        temp[k++] = arr[i++];
    }
    
    while (j <= right) {
        temp[k++] = arr[j++];
    }
    
    for (i = left; i <= right; i++) {
        arr[i] = temp[i];
    }
}

void parallel_merge_sort(int arr[], int n) {
    int *temp = (int*)malloc(n * sizeof(int));
    
    // Bottom-up merge sort (simulates parallel execution)
    for (int size = 1; size < n; size *= 2) {
        // Process independent merge operations
        for (int left = 0; left < n - size; left += 2 * size) {
            int mid = left + size - 1;
            int right = (left + 2 * size - 1 < n - 1) ? left + 2 * size - 1 : n - 1;
            merge(arr, left, mid, right, temp);
        }
    }
    
    free(temp);
}

void generate_array(int arr[], int n) {
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        arr[i] = seed % 1000000;
    }
}

int verify_sorted(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) return 0;
    }
    return 1;
}

int main() {
    int *arr = (int*)malloc(ARRAY_SIZE * sizeof(int));
    int *original = (int*)malloc(ARRAY_SIZE * sizeof(int));
    
    generate_array(original, ARRAY_SIZE);
    
    clock_t start = clock();
    
    int sorted_count = 0;
    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        memcpy(arr, original, ARRAY_SIZE * sizeof(int));
        parallel_merge_sort(arr, ARRAY_SIZE);
        if (verify_sorted(arr, ARRAY_SIZE)) {
            sorted_count++;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Parallel merge sort: array=%d, %d trials, %.6f seconds\n",
           ARRAY_SIZE, NUM_TRIALS, time_spent);
    printf("Successfully sorted: %d/%d\n", sorted_count, NUM_TRIALS);
    
    free(arr);
    free(original);
    
    return 0;
}
