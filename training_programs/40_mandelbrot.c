// Mandelbrot set computation
// Complex arithmetic, nested loops, divergence testing
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 512
#define HEIGHT 512
#define MAX_ITER 256

int mandelbrot_iter(double cx, double cy) {
    double zx = 0.0, zy = 0.0;
    int iter = 0;
    
    while (zx * zx + zy * zy < 4.0 && iter < MAX_ITER) {
        double temp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = temp;
        iter++;
    }
    
    return iter;
}

void compute_mandelbrot(int *output, int width, int height) {
    double x_min = -2.5, x_max = 1.0;
    double y_min = -1.0, y_max = 1.0;
    
    for (int py = 0; py < height; py++) {
        for (int px = 0; px < width; px++) {
            double cx = x_min + (x_max - x_min) * px / width;
            double cy = y_min + (y_max - y_min) * py / height;
            output[py * width + px] = mandelbrot_iter(cx, cy);
        }
    }
}

int main() {
    int *output = (int*)malloc(WIDTH * HEIGHT * sizeof(int));
    
    clock_t start = clock();
    compute_mandelbrot(output, WIDTH, HEIGHT);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    long long total = 0;
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        total += output[i];
    }
    
    printf("Mandelbrot %dx%d: %.6f seconds, avg_iter=%.2f\n",
           WIDTH, HEIGHT, time_spent, (double)total / (WIDTH * HEIGHT));
    
    free(output);
    return 0;
}
