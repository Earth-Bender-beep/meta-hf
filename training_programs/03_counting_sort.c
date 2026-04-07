#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void counting_sort(int arr[], int n, int max_val) {
    int* count = (int*)calloc(max_val + 1, sizeof(int));
    int* output = (int*)malloc(n * sizeof(int));
    
    for (int i = 0; i < n; i++)
        count[arr[i]]++;
    
    for (int i = 1; i <= max_val; i++)
        count[i] += count[i - 1];
    
    for (int i = n - 1; i >= 0; i--) {
        output[count[arr[i]] - 1] = arr[i];
        count[arr[i]]--;
    }
    
    for (int i = 0; i < n; i++)
        arr[i] = output[i];
    
    free(count);
    free(output);
}

int main() {
    int n = 100000;
    int max_val = 10000;
    int* arr = (int*)malloc(n * sizeof(int));
    
    srand(42);
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % (max_val + 1);
    }
    
    clock_t start = clock();
    counting_sort(arr, n, max_val);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Counting sort: %d elements in %.6f seconds\n", n, time_spent);
    
    free(arr);
    return 0;
}
