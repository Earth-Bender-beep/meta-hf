// Watershed segmentation algorithm for image processing
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

#define IMAGE_SIZE 128
#define WATERSHED_MARK -1
#define INIT -2

typedef struct {
    int x, y;
} Point;

typedef struct {
    Point *data;
    int size;
    int capacity;
} Queue;

Queue* create_queue(int capacity) {
    Queue *q = (Queue*)malloc(sizeof(Queue));
    q->data = (Point*)malloc(capacity * sizeof(Point));
    q->size = 0;
    q->capacity = capacity;
    return q;
}

void enqueue(Queue *q, int x, int y) {
    if (q->size < q->capacity) {
        q->data[q->size].x = x;
        q->data[q->size].y = y;
        q->size++;
    }
}

Point dequeue(Queue *q) {
    Point p = q->data[0];
    for (int i = 0; i < q->size - 1; i++) {
        q->data[i] = q->data[i + 1];
    }
    q->size--;
    return p;
}

void watershed_segmentation(unsigned char **image, int **labels, int width, int height) {
    Queue *queue = create_queue(width * height);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            labels[y][x] = INIT;
        }
    }
    
    int current_label = 0;
    
    for (int intensity = 0; intensity < 256; intensity++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (image[y][x] == intensity && labels[y][x] == INIT) {
                    int dirs[4][2] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
                    int found_label = 0;
                    
                    for (int d = 0; d < 4; d++) {
                        int nx = x + dirs[d][0];
                        int ny = y + dirs[d][1];
                        
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            if (labels[ny][nx] > 0) {
                                found_label = labels[ny][nx];
                                break;
                            }
                        }
                    }
                    
                    if (found_label > 0) {
                        labels[y][x] = found_label;
                        enqueue(queue, x, y);
                    } else {
                        labels[y][x] = ++current_label;
                        enqueue(queue, x, y);
                    }
                }
            }
        }
        
        while (queue->size > 0) {
            Point p = dequeue(queue);
            int dirs[4][2] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
            
            for (int d = 0; d < 4; d++) {
                int nx = p.x + dirs[d][0];
                int ny = p.y + dirs[d][1];
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    if (labels[ny][nx] == INIT && image[ny][nx] == intensity) {
                        labels[ny][nx] = labels[p.y][p.x];
                        enqueue(queue, nx, ny);
                    }
                }
            }
        }
    }
    
    free(queue->data);
    free(queue);
}

int main() {
    int size = IMAGE_SIZE;
    
    unsigned char **image = (unsigned char**)malloc(size * sizeof(unsigned char*));
    int **labels = (int**)malloc(size * sizeof(int*));
    for (int i = 0; i < size; i++) {
        image[i] = (unsigned char*)malloc(size);
        labels[i] = (int*)malloc(size * sizeof(int));
    }
    
    unsigned int seed = 42;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            seed = seed * 1103515245 + 12345;
            image[y][x] = (seed & 0xFF);
        }
    }
    
    clock_t start = clock();
    watershed_segmentation(image, labels, size, size);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    int num_segments = 0;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            if (labels[y][x] > num_segments) {
                num_segments = labels[y][x];
            }
        }
    }
    
    printf("Watershed segmentation: %dx%d image, %.6f seconds\n", size, size, time_spent);
    printf("Number of segments: %d\n", num_segments);
    
    for (int i = 0; i < size; i++) {
        free(image[i]);
        free(labels[i]);
    }
    free(image);
    free(labels);
    
    return 0;
}
