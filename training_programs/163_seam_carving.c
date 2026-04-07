// Seam carving for content-aware image resizing
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define IMAGE_WIDTH 200
#define IMAGE_HEIGHT 200
#define NUM_SEAMS 20

int energy(unsigned char *image, int width, int height, int x, int y) {
    int dx = 0, dy = 0;
    
    if (x > 0 && x < width - 1) {
        dx = abs(image[y * width + x + 1] - image[y * width + x - 1]);
    }
    
    if (y > 0 && y < height - 1) {
        dy = abs(image[(y + 1) * width + x] - image[(y - 1) * width + x]);
    }
    
    return dx + dy;
}

void compute_energy_map(unsigned char *image, int *energy_map, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            energy_map[y * width + x] = energy(image, width, height, x, y);
        }
    }
}

void find_vertical_seam(int *energy_map, int width, int height, int *seam) {
    int *dp = (int*)malloc(width * height * sizeof(int));
    
    for (int x = 0; x < width; x++) {
        dp[x] = energy_map[x];
    }
    
    for (int y = 1; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int min_prev = dp[(y - 1) * width + x];
            
            if (x > 0) {
                int left = dp[(y - 1) * width + x - 1];
                if (left < min_prev) min_prev = left;
            }
            
            if (x < width - 1) {
                int right = dp[(y - 1) * width + x + 1];
                if (right < min_prev) min_prev = right;
            }
            
            dp[y * width + x] = energy_map[y * width + x] + min_prev;
        }
    }
    
    int min_x = 0;
    for (int x = 1; x < width; x++) {
        if (dp[(height - 1) * width + x] < dp[(height - 1) * width + min_x]) {
            min_x = x;
        }
    }
    
    seam[height - 1] = min_x;
    
    for (int y = height - 2; y >= 0; y--) {
        int x = seam[y + 1];
        int min_x = x;
        int min_energy = dp[y * width + x];
        
        if (x > 0 && dp[y * width + x - 1] < min_energy) {
            min_x = x - 1;
            min_energy = dp[y * width + x - 1];
        }
        
        if (x < width - 1 && dp[y * width + x + 1] < min_energy) {
            min_x = x + 1;
        }
        
        seam[y] = min_x;
    }
    
    free(dp);
}

int main() {
    int width = IMAGE_WIDTH;
    int height = IMAGE_HEIGHT;
    
    unsigned char *image = (unsigned char*)malloc(width * height);
    int *energy_map = (int*)malloc(width * height * sizeof(int));
    int *seam = (int*)malloc(height * sizeof(int));
    
    unsigned int seed = 42;
    for (int i = 0; i < width * height; i++) {
        seed = seed * 1103515245 + 12345;
        image[i] = seed & 0xFF;
    }
    
    clock_t start = clock();
    
    for (int s = 0; s < NUM_SEAMS; s++) {
        compute_energy_map(image, energy_map, width, height);
        find_vertical_seam(energy_map, width, height, seam);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Seam carving: %dx%d image, %d seams, %.6f seconds\n",
           width, height, NUM_SEAMS, time_spent);
    
    free(image);
    free(energy_map);
    free(seam);
    
    return 0;
}
