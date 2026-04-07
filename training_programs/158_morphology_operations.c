// Morphological operations - dilation and erosion for image processing
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define IMAGE_SIZE 256
#define KERNEL_SIZE 5

void dilate(unsigned char *input, unsigned char *output, int width, int height) {
    int half = KERNEL_SIZE / 2;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned char max_val = 0;
            
            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;
                    
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        unsigned char val = input[ny * width + nx];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
            }
            
            output[y * width + x] = max_val;
        }
    }
}

void erode(unsigned char *input, unsigned char *output, int width, int height) {
    int half = KERNEL_SIZE / 2;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned char min_val = 255;
            
            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;
                    
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        unsigned char val = input[ny * width + nx];
                        if (val < min_val) {
                            min_val = val;
                        }
                    }
                }
            }
            
            output[y * width + x] = min_val;
        }
    }
}

void opening(unsigned char *input, unsigned char *output, unsigned char *temp, int width, int height) {
    erode(input, temp, width, height);
    dilate(temp, output, width, height);
}

void closing(unsigned char *input, unsigned char *output, unsigned char *temp, int width, int height) {
    dilate(input, temp, width, height);
    erode(temp, output, width, height);
}

int main() {
    int size = IMAGE_SIZE;
    unsigned char *image = (unsigned char*)malloc(size * size);
    unsigned char *dilated = (unsigned char*)malloc(size * size);
    unsigned char *eroded = (unsigned char*)malloc(size * size);
    unsigned char *temp = (unsigned char*)malloc(size * size);
    
    unsigned int seed = 42;
    for (int i = 0; i < size * size; i++) {
        seed = seed * 1103515245 + 12345;
        image[i] = (seed & 0xFF);
    }
    
    clock_t start = clock();
    
    dilate(image, dilated, size, size);
    erode(image, eroded, size, size);
    opening(image, temp, dilated, size, size);
    closing(image, temp, eroded, size, size);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Morphological operations: %dx%d image, %.6f seconds\n", 
           size, size, time_spent);
    
    free(image);
    free(dilated);
    free(eroded);
    free(temp);
    
    return 0;
}
