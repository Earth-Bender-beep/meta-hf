// Integral image (summed area table) for fast region sum queries
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define IMAGE_SIZE 512
#define NUM_QUERIES 10000

void compute_integral_image(unsigned char *image, long long *integral, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            long long val = image[y * width + x];
            long long left = (x > 0) ? integral[y * width + x - 1] : 0;
            long long top = (y > 0) ? integral[(y - 1) * width + x] : 0;
            long long diag = (x > 0 && y > 0) ? integral[(y - 1) * width + x - 1] : 0;
            
            integral[y * width + x] = val + left + top - diag;
        }
    }
}

long long query_sum(long long *integral, int width, int x1, int y1, int x2, int y2) {
    long long total = integral[y2 * width + x2];
    long long left = (x1 > 0) ? integral[y2 * width + x1 - 1] : 0;
    long long top = (y1 > 0) ? integral[(y1 - 1) * width + x2] : 0;
    long long diag = (x1 > 0 && y1 > 0) ? integral[(y1 - 1) * width + x1 - 1] : 0;
    
    return total - left - top + diag;
}

int main() {
    int size = IMAGE_SIZE;
    unsigned char *image = (unsigned char*)malloc(size * size);
    long long *integral = (long long*)malloc(size * size * sizeof(long long));
    
    unsigned int seed = 42;
    for (int i = 0; i < size * size; i++) {
        seed = seed * 1103515245 + 12345;
        image[i] = seed & 0xFF;
    }
    
    clock_t start = clock();
    
    compute_integral_image(image, integral, size, size);
    
    long long total_sum = 0;
    for (int q = 0; q < NUM_QUERIES; q++) {
        seed = seed * 1103515245 + 12345;
        int x1 = seed % (size - 50);
        seed = seed * 1103515245 + 12345;
        int y1 = seed % (size - 50);
        int x2 = x1 + 50;
        int y2 = y1 + 50;
        
        total_sum += query_sum(integral, size, x1, y1, x2, y2);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Integral image: %dx%d image, %d queries, %.6f seconds\n",
           size, size, NUM_QUERIES, time_spent);
    printf("Total sum: %lld\n", total_sum);
    
    free(image);
    free(integral);
    
    return 0;
}
