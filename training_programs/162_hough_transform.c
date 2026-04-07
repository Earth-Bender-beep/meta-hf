// Hough transform for line detection in images
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define IMAGE_SIZE 256
#define NUM_THETA 180
#define NUM_RHO 360

void hough_transform(unsigned char *edges, int width, int height, int *accumulator) {
    double max_rho = sqrt(width * width + height * height);
    
    for (int i = 0; i < NUM_THETA * NUM_RHO; i++) {
        accumulator[i] = 0;
    }
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (edges[y * width + x] > 128) {
                for (int theta_idx = 0; theta_idx < NUM_THETA; theta_idx++) {
                    double theta = (theta_idx * M_PI) / NUM_THETA;
                    double rho = x * cos(theta) + y * sin(theta);
                    
                    int rho_idx = (int)((rho + max_rho) * NUM_RHO / (2 * max_rho));
                    
                    if (rho_idx >= 0 && rho_idx < NUM_RHO) {
                        accumulator[theta_idx * NUM_RHO + rho_idx]++;
                    }
                }
            }
        }
    }
}

void find_peaks(int *accumulator, int threshold, int *num_lines) {
    *num_lines = 0;
    
    for (int i = 0; i < NUM_THETA * NUM_RHO; i++) {
        if (accumulator[i] > threshold) {
            (*num_lines)++;
        }
    }
}

int main() {
    int size = IMAGE_SIZE;
    unsigned char *edges = (unsigned char*)malloc(size * size);
    int *accumulator = (int*)malloc(NUM_THETA * NUM_RHO * sizeof(int));
    
    unsigned int seed = 42;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            seed = seed * 1103515245 + 12345;
            
            if (abs(y - size/2) < 2 || abs(x - size/2) < 2) {
                edges[y * size + x] = 255;
            } else {
                edges[y * size + x] = (seed & 0xFF) > 250 ? 255 : 0;
            }
        }
    }
    
    clock_t start = clock();
    
    hough_transform(edges, size, size, accumulator);
    
    int num_lines;
    find_peaks(accumulator, 50, &num_lines);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Hough transform: %dx%d image, %.6f seconds\n", size, size, time_spent);
    printf("Detected %d potential lines\n", num_lines);
    
    free(edges);
    free(accumulator);
    
    return 0;
}
