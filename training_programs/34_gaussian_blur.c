// Gaussian blur filter for image processing
// Array-heavy, moderate branching, good vectorization opportunity
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define WIDTH 256
#define HEIGHT 256
#define KERNEL_SIZE 5

typedef struct {
    unsigned char data[HEIGHT][WIDTH];
} Image;

void generate_gaussian_kernel(float kernel[KERNEL_SIZE][KERNEL_SIZE], float sigma) {
    float sum = 0.0;
    int half = KERNEL_SIZE / 2;
    
    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float value = exp(-(x*x + y*y) / (2.0 * sigma * sigma));
            kernel[y + half][x + half] = value;
            sum += value;
        }
    }
    
    // Normalize
    for (int y = 0; y < KERNEL_SIZE; y++) {
        for (int x = 0; x < KERNEL_SIZE; x++) {
            kernel[y][x] /= sum;
        }
    }
}

void apply_gaussian_blur(Image *input, Image *output, float kernel[KERNEL_SIZE][KERNEL_SIZE]) {
    int half = KERNEL_SIZE / 2;
    
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            float sum = 0.0;
            
            for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                    int py = y + ky - half;
                    int px = x + kx - half;
                    
                    // Clamp to borders
                    if (py < 0) py = 0;
                    if (py >= HEIGHT) py = HEIGHT - 1;
                    if (px < 0) px = 0;
                    if (px >= WIDTH) px = WIDTH - 1;
                    
                    sum += input->data[py][px] * kernel[ky][kx];
                }
            }
            
            output->data[y][x] = (unsigned char)(sum + 0.5);
        }
    }
}

void init_test_image(Image *img) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            img->data[y][x] = (x + y) % 256;
        }
    }
}

int main() {
    Image *input = (Image*)malloc(sizeof(Image));
    Image *output = (Image*)malloc(sizeof(Image));
    float kernel[KERNEL_SIZE][KERNEL_SIZE];
    
    init_test_image(input);
    generate_gaussian_kernel(kernel, 1.4);
    
    clock_t start = clock();
    apply_gaussian_blur(input, output, kernel);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Gaussian blur %dx%d (kernel=%d): %.6f seconds, pixel[128][128]=%d\n",
           WIDTH, HEIGHT, KERNEL_SIZE, time_spent, output->data[128][128]);
    
    free(input);
    free(output);
    return 0;
}
