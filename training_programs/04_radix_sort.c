#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int get_max(int arr[], int n) {
    int max = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > max)
            max = arr[i];
    return max;
}

void counting_sort_by_digit(int arr[], int n, int exp) {
    int* output = (int*)malloc(n * sizeof(int));
    int count[10] = {0};
    
    for (int i = 0; i < n; i++)
        count[(arr[i] / exp) % 10]++;
    
    for (int i = 1; i < 10; i++)
        count[i] += count[i - 1];
    
    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }
    
    for (int i = 0; i < n; i++)
        arr[i] = output[i];
    
    free(output);
}

void radix_sort(int arr[], int n) {
    int max = get_max(arr, n);
    
    for (int exp = 1; max / exp > 0; exp *= 10)
        counting_sort_by_digit(arr, n, exp);
}

int main() {
    int n = 100000;
    int* arr = (int*)malloc(n * sizeof(int));
    
    srand(42);
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000000;
    }
    
    clock_t start = clock();
    radix_sort(arr, n);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Radix sort: %d elements in %.6f seconds\n", n, time_spent);
    
    free(arr);
    return 0;
}
