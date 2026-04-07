#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct MinHeap {
    int *arr;
    int size;
    int capacity;
} MinHeap;

MinHeap* createMinHeap(int capacity) {
    MinHeap* heap = (MinHeap*)malloc(sizeof(MinHeap));
    heap->arr = (int*)malloc(capacity * sizeof(int));
    heap->size = 0;
    heap->capacity = capacity;
    return heap;
}

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void minHeapify(MinHeap* heap, int idx) {
    int smallest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
    
    if (left < heap->size && heap->arr[left] < heap->arr[smallest])
        smallest = left;
    if (right < heap->size && heap->arr[right] < heap->arr[smallest])
        smallest = right;
    
    if (smallest != idx) {
        swap(&heap->arr[idx], &heap->arr[smallest]);
        minHeapify(heap, smallest);
    }
}

void insert(MinHeap* heap, int key) {
    if (heap->size == heap->capacity)
        return;
    
    heap->size++;
    int i = heap->size - 1;
    heap->arr[i] = key;
    
    while (i != 0 && heap->arr[(i - 1) / 2] > heap->arr[i]) {
        swap(&heap->arr[i], &heap->arr[(i - 1) / 2]);
        i = (i - 1) / 2;
    }
}

int extractMin(MinHeap* heap) {
    if (heap->size <= 0)
        return -1;
    if (heap->size == 1) {
        heap->size--;
        return heap->arr[0];
    }
    
    int root = heap->arr[0];
    heap->arr[0] = heap->arr[heap->size - 1];
    heap->size--;
    minHeapify(heap, 0);
    
    return root;
}

int main() {
    int n = 100000;
    MinHeap* heap = createMinHeap(n);
    
    srand(42);
    clock_t start = clock();
    
    for (int i = 0; i < n; i++) {
        insert(heap, rand() % 100000);
    }
    
    for (int i = 0; i < n / 2; i++) {
        extractMin(heap);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Min heap: %d ops in %.6f seconds\n", n + n/2, time_spent);
    
    free(heap->arr);
    free(heap);
    return 0;
}
