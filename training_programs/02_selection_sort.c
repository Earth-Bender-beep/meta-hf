#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void selection_sort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_idx])
                min_idx = j;
        }
        
        int temp = arr[min_idx];
        arr[min_idx] = arr[i];
        arr[i] = temp;
    }
}

int main() {
    int n = 20000;
    int* arr = (int*)malloc(n * sizeof(int));
    
    srand(42);
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 50000;
    }
    
    clock_t start = clock();
    selection_sort(arr, n);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Selection sort: %d elements in %.6f seconds\n", n, time_spent);
    
    free(arr);
    return 0;
}
