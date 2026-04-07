// Histogram equalization for image contrast enhancement
// Cumulative distribution function computation
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 256
#define HEIGHT 256
#define LEVELS 256

typedef struct {
    unsigned char data[HEIGHT][WIDTH];
} Image;

void compute_histogram(Image *img, int *histogram) {
    for (int i = 0; i < LEVELS; i++) {
        histogram[i] = 0;
    }
    
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            histogram[img->data[y][x]]++;
        }
    }
}

void histogram_equalization(Image *input, Image *output) {
    int histogram[LEVELS];
    int cdf[LEVELS];
    
    // Compute histogram
    compute_histogram(input, histogram);
    
    // Compute cumulative distribution function
    cdf[0] = histogram[0];
    for (int i = 1; i < LEVELS; i++) {
        cdf[i] = cdf[i-1] + histogram[i];
    }
    
    // Find minimum non-zero CDF value
    int cdf_min = cdf[0];
    for (int i = 0; i < LEVELS; i++) {
        if (cdf[i] > 0) {
            cdf_min = cdf[i];
            break;
        }
    }
    
    // Create lookup table for equalization
    unsigned char lut[LEVELS];
    int total_pixels = WIDTH * HEIGHT;
    
    for (int i = 0; i < LEVELS; i++) {
        if (cdf[i] == 0) {
            lut[i] = 0;
        } else {
            lut[i] = (unsigned char)(((cdf[i] - cdf_min) * (LEVELS - 1)) / 
                                     (total_pixels - cdf_min));
        }
    }
    
    // Apply equalization
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            output->data[y][x] = lut[input->data[y][x]];
        }
    }
}

void init_low_contrast_image(Image *img) {
    // Create image with poor contrast (values clustered in narrow range)
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int base = 80 + ((x + y) % 60);
            img->data[y][x] = (unsigned char)base;
        }
    }
}

double compute_contrast(Image *img) {
    int histogram[LEVELS];
    compute_histogram(img, histogram);
    
    // Compute mean
    double mean = 0.0;
    for (int i = 0; i < LEVELS; i++) {
        mean += i * histogram[i];
    }
    mean /= (WIDTH * HEIGHT);
    
    // Compute variance (measure of contrast)
    double variance = 0.0;
    for (int i = 0; i < LEVELS; i++) {
        double diff = i - mean;
        variance += diff * diff * histogram[i];
    }
    variance /= (WIDTH * HEIGHT);
    
    return variance;
}

int main() {
    Image *input = (Image*)malloc(sizeof(Image));
    Image *output = (Image*)malloc(sizeof(Image));
    
    init_low_contrast_image(input);
    
    double contrast_before = compute_contrast(input);
    
    clock_t start = clock();
    histogram_equalization(input, output);
    clock_t end = clock();
    
    double contrast_after = compute_contrast(output);
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Histogram equalization %dx%d: %.6f seconds\n", WIDTH, HEIGHT, time_spent);
    printf("Contrast: before=%.2f, after=%.2f (%.1f%% improvement)\n",
           contrast_before, contrast_after, 
           100.0 * (contrast_after - contrast_before) / contrast_before);
    
    free(input);
    free(output);
    return 0;
}
