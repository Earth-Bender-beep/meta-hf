// Ternary heap - heap with 3 children per node
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define HEAP_SIZE 10000

typedef struct {
    int *data;
    int size;
    int capacity;
} TernaryHeap;

TernaryHeap* create_ternary_heap(int capacity) {
    TernaryHeap *h = (TernaryHeap*)malloc(sizeof(TernaryHeap));
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

void heapify_down(TernaryHeap *h, int idx) {
    int largest = idx;
    
    for (int i = 1; i <= 3; i++) {
        int child = 3 * idx + i;
        if (child < h->size && h->data[child] > h->data[largest]) {
            largest = child;
        }
    }
    
    if (largest != idx) {
        swap(&h->data[idx], &h->data[largest]);
        heapify_down(h, largest);
    }
}

void heapify_up(TernaryHeap *h, int idx) {
    if (idx == 0) return;
    
    int parent = (idx - 1) / 3;
    if (h->data[idx] > h->data[parent]) {
        swap(&h->data[idx], &h->data[parent]);
        heapify_up(h, parent);
    }
}

void insert(TernaryHeap *h, int value) {
    if (h->size >= h->capacity) return;
    
    h->data[h->size] = value;
    heapify_up(h, h->size);
    h->size++;
}

int extract_max(TernaryHeap *h) {
    if (h->size == 0) return -1;
    
    int max = h->data[0];
    h->data[0] = h->data[h->size - 1];
    h->size--;
    heapify_down(h, 0);
    
    return max;
}

int main() {
    TernaryHeap *h = create_ternary_heap(HEAP_SIZE);
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    for (int i = 0; i < HEAP_SIZE; i++) {
        seed = seed * 1103515245 + 12345;
        insert(h, seed % 100000);
    }
    
    long long sum = 0;
    for (int i = 0; i < HEAP_SIZE / 2; i++) {
        sum += extract_max(h);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Ternary heap: %d operations, %.6f seconds\n", HEAP_SIZE, time_spent);
    printf("Sum: %lld\n", sum);
    
    free(h->data);
    free(h);
    
    return 0;
}
