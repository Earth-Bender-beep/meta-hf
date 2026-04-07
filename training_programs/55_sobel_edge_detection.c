// Sobel edge detection filter
// Image gradient computation with convolution kernels
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define WIDTH 256
#define HEIGHT 256

typedef struct {
    unsigned char data[HEIGHT][WIDTH];
} Image;

void sobel_filter(Image *input, Image *output) {
    // Sobel kernels for gradient computation
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    for (int y = 1; y < HEIGHT - 1; y++) {
        for (int x = 1; x < WIDTH - 1; x++) {
            int sum_x = 0, sum_y = 0;
            
            // Apply kernels
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel = input->data[y + ky][x + kx];
                    sum_x += pixel * gx[ky + 1][kx + 1];
                    sum_y += pixel * gy[ky + 1][kx + 1];
                }
            }
            
            // Gradient magnitude
            int magnitude = (int)sqrt(sum_x * sum_x + sum_y * sum_y);
            output->data[y][x] = (magnitude > 255) ? 255 : (unsigned char)magnitude;
        }
    }
    
    // Handle borders
    for (int x = 0; x < WIDTH; x++) {
        output->data[0][x] = 0;
        output->data[HEIGHT-1][x] = 0;
    }
    for (int y = 0; y < HEIGHT; y++) {
        output->data[y][0] = 0;
        output->data[y][WIDTH-1] = 0;
    }
}

void init_test_image(Image *img) {
    // Create a test pattern with edges
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (x < WIDTH / 2) {
                img->data[y][x] = 50;
            } else {
                img->data[y][x] = 200;
            }
            
            // Add diagonal pattern
            if ((x + y) % 40 < 5) {
                img->data[y][x] = 255;
            }
        }
    }
}

int main() {
    Image *input = (Image*)malloc(sizeof(Image));
    Image *output = (Image*)malloc(sizeof(Image));
    
    init_test_image(input);
    
    clock_t start = clock();
    sobel_filter(input, output);
    clock_t end = clock();
    
    // Count edge pixels
    int edge_count = 0;
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (output->data[y][x] > 50) {
                edge_count++;
            }
        }
    }
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Sobel edge detection %dx%d: %.6f seconds, %d edge pixels\n",
           WIDTH, HEIGHT, time_spent, edge_count);
    
    free(input);
    free(output);
    return 0;
}
