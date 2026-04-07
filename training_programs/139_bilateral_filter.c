// Bilateral filter - edge-preserving image smoothing
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define IMAGE_SIZE 256
#define WINDOW_SIZE 5

double gaussian(double x, double sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

void bilateral_filter(double *input, double *output, int width, int height, 
                     double sigma_spatial, double sigma_range) {
    int half_window = WINDOW_SIZE / 2;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double sum = 0.0;
            double weight_sum = 0.0;
            double center_value = input[y * width + x];
            
            for (int dy = -half_window; dy <= half_window; dy++) {
                for (int dx = -half_window; dx <= half_window; dx++) {
                    int ny = y + dy;
                    int nx = x + dx;
                    
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        double neighbor_value = input[ny * width + nx];
                        
                        // Spatial weight
                        double spatial_dist = sqrt(dx * dx + dy * dy);
                        double spatial_weight = gaussian(spatial_dist, sigma_spatial);
                        
                        // Range weight
                        double range_dist = neighbor_value - center_value;
                        double range_weight = gaussian(range_dist, sigma_range);
                        
                        double weight = spatial_weight * range_weight;
                        
                        sum += neighbor_value * weight;
                        weight_sum += weight;
                    }
                }
            }
            
            output[y * width + x] = sum / weight_sum;
        }
    }
}

int main() {
    int size = IMAGE_SIZE;
    double *image = (double*)malloc(size * size * sizeof(double));
    double *filtered = (double*)malloc(size * size * sizeof(double));
    
    unsigned int seed = 42;
    for (int i = 0; i < size * size; i++) {
        seed = seed * 1103515245 + 12345;
        image[i] = (seed & 0xFF) / 255.0;
    }
    
    clock_t start = clock();
    bilateral_filter(image, filtered, size, size, 2.0, 0.1);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Bilateral filter: %dx%d image, %.6f seconds\n", size, size, time_spent);
    
    free(image);
    free(filtered);
    
    return 0;
}
