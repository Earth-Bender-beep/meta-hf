#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void insertion_sort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

int main() {
    int n = 30000;
    int* arr = (int*)malloc(n * sizeof(int));
    
    srand(42);
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 50000;
    }
    
    clock_t start = clock();
    insertion_sort(arr, n);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Insertion sort: %d elements in %.6f seconds\n", n, time_spent);
    
    free(arr);
    return 0;
}
