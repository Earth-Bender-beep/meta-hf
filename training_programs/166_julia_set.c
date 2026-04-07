// Julia set fractal generator
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 512
#define HEIGHT 512
#define MAX_ITER 256

int julia_iteration(double zx, double zy, double cx, double cy, int max_iter) {
    int iter = 0;
    
    while (zx * zx + zy * zy < 4.0 && iter < max_iter) {
        double temp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = temp;
        iter++;
    }
    
    return iter;
}

void generate_julia_set(unsigned char *image, int width, int height,
                       double cx, double cy, int max_iter) {
    double x_min = -2.0, x_max = 2.0;
    double y_min = -2.0, y_max = 2.0;
    
    for (int py = 0; py < height; py++) {
        for (int px = 0; px < width; px++) {
            double zx = x_min + (px / (double)width) * (x_max - x_min);
            double zy = y_min + (py / (double)height) * (y_max - y_min);
            
            int iter = julia_iteration(zx, zy, cx, cy, max_iter);
            
            image[py * width + px] = (unsigned char)((iter * 255) / max_iter);
        }
    }
}

int main() {
    int width = WIDTH;
    int height = HEIGHT;
    
    unsigned char *image = (unsigned char*)malloc(width * height);
    
    double cx = -0.7;
    double cy = 0.27015;
    
    clock_t start = clock();
    
    generate_julia_set(image, width, height, cx, cy, MAX_ITER);
    
    long long checksum = 0;
    for (int i = 0; i < width * height; i++) {
        checksum += image[i];
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Julia set: %dx%d, max_iter=%d, %.6f seconds\n",
           width, height, MAX_ITER, time_spent);
    printf("Checksum: %lld\n", checksum);
    
    free(image);
    
    return 0;
}
