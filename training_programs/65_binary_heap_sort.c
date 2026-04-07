// Binary heap implementation with heap sort
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define HEAP_SIZE 10000

typedef struct {
    int *data;
    int size;
    int capacity;
} Heap;

Heap* create_heap(int capacity) {
    Heap *h = (Heap*)malloc(sizeof(Heap));
    h->data = (int*)malloc(capacity * sizeof(int));
    h->size = 0;
    h->capacity = capacity;
    return h;
}

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void heapify_down(Heap *h, int idx) {
    int largest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
    
    if (left < h->size && h->data[left] > h->data[largest]) {
        largest = left;
    }
    
    if (right < h->size && h->data[right] > h->data[largest]) {
        largest = right;
    }
    
    if (largest != idx) {
        swap(&h->data[idx], &h->data[largest]);
        heapify_down(h, largest);
    }
}

void heapify_up(Heap *h, int idx) {
    if (idx == 0) return;
    
    int parent = (idx - 1) / 2;
    if (h->data[idx] > h->data[parent]) {
        swap(&h->data[idx], &h->data[parent]);
        heapify_up(h, parent);
    }
}

void heap_insert(Heap *h, int value) {
    if (h->size >= h->capacity) return;
    
    h->data[h->size] = value;
    heapify_up(h, h->size);
    h->size++;
}

int heap_extract_max(Heap *h) {
    if (h->size == 0) return -1;
    
    int max = h->data[0];
    h->data[0] = h->data[h->size - 1];
    h->size--;
    heapify_down(h, 0);
    
    return max;
}

void heap_sort(int *arr, int n) {
    Heap *h = create_heap(n);
    
    for (int i = 0; i < n; i++) {
        heap_insert(h, arr[i]);
    }
    
    for (int i = n - 1; i >= 0; i--) {
        arr[i] = heap_extract_max(h);
    }
    
    free(h->data);
    free(h);
}

int main() {
    int *arr = (int*)malloc(HEAP_SIZE * sizeof(int));
    unsigned int seed = 42;
    
    for (int i = 0; i < HEAP_SIZE; i++) {
        seed = seed * 1103515245 + 12345;
        arr[i] = seed % 100000;
    }
    
    clock_t start = clock();
    heap_sort(arr, HEAP_SIZE);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Heap sort: %d elements, %.6f seconds\n", HEAP_SIZE, time_spent);
    
    free(arr);
    return 0;
}
