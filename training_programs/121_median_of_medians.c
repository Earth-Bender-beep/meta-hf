// Median of medians algorithm for linear-time selection (finding k-th smallest element)
// Deterministic O(n) selection algorithm
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE 50000
#define NUM_TESTS 100

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void insertion_sort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

int partition(int arr[], int left, int right, int pivot) {
    // Find pivot and move to end
    for (int i = left; i <= right; i++) {
        if (arr[i] == pivot) {
            swap(&arr[i], &arr[right]);
            break;
        }
    }
    
    int i = left;
    for (int j = left; j < right; j++) {
        if (arr[j] < pivot) {
            swap(&arr[i], &arr[j]);
            i++;
        }
    }
    swap(&arr[i], &arr[right]);
    return i;
}

int select_pivot(int arr[], int left, int right) {
    if (right - left < 5) {
        insertion_sort(arr + left, right - left + 1);
        return arr[left + (right - left) / 2];
    }
    
    // Divide into groups of 5
    int num_medians = 0;
    for (int i = left; i <= right; i += 5) {
        int sub_right = (i + 4 <= right) ? i + 4 : right;
        insertion_sort(arr + i, sub_right - i + 1);
        int median = arr[i + (sub_right - i) / 2];
        swap(&arr[left + num_medians], &arr[i + (sub_right - i) / 2]);
        num_medians++;
    }
    
    // Recursively find median of medians
    return select_pivot(arr, left, left + num_medians - 1);
}

int median_of_medians(int arr[], int left, int right, int k) {
    if (left == right) {
        return arr[left];
    }
    
    int pivot = select_pivot(arr, left, right);
    int pivot_index = partition(arr, left, right, pivot);
    
    if (k == pivot_index) {
        return arr[k];
    } else if (k < pivot_index) {
        return median_of_medians(arr, left, pivot_index - 1, k);
    } else {
        return median_of_medians(arr, pivot_index + 1, right, k);
    }
}

void generate_array(int arr[], int size) {
    unsigned int seed = 42;
    for (int i = 0; i < size; i++) {
        seed = seed * 1103515245 + 12345;
        arr[i] = seed % 100000;
    }
}

int main() {
    int *arr = (int*)malloc(ARRAY_SIZE * sizeof(int));
    int *temp = (int*)malloc(ARRAY_SIZE * sizeof(int));
    
    generate_array(arr, ARRAY_SIZE);
    
    clock_t start = clock();
    
    long long sum = 0;
    for (int test = 0; test < NUM_TESTS; test++) {
        // Copy array for each test
        for (int i = 0; i < ARRAY_SIZE; i++) {
            temp[i] = arr[i];
        }
        
        // Find median
        int median = median_of_medians(temp, 0, ARRAY_SIZE - 1, ARRAY_SIZE / 2);
        sum += median;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Median of medians: array=%d, %d tests, %.6f seconds\n",
           ARRAY_SIZE, NUM_TESTS, time_spent);
    printf("Average median: %lld\n", sum / NUM_TESTS);
    
    free(arr);
    free(temp);
    
    return 0;
}
