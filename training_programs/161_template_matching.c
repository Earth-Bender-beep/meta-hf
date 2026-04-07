// Template matching using normalized cross-correlation
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define IMAGE_WIDTH 256
#define IMAGE_HEIGHT 256
#define TEMPLATE_SIZE 16

double normalized_cross_correlation(unsigned char *image, unsigned char *templ,
                                   int img_width, int x, int y, int templ_size) {
    double sum_img = 0.0, sum_templ = 0.0;
    double sum_img_sq = 0.0, sum_templ_sq = 0.0;
    double sum_product = 0.0;
    int count = templ_size * templ_size;
    
    for (int ty = 0; ty < templ_size; ty++) {
        for (int tx = 0; tx < templ_size; tx++) {
            double img_val = image[(y + ty) * img_width + (x + tx)];
            double templ_val = templ[ty * templ_size + tx];
            
            sum_img += img_val;
            sum_templ += templ_val;
            sum_img_sq += img_val * img_val;
            sum_templ_sq += templ_val * templ_val;
            sum_product += img_val * templ_val;
        }
    }
    
    double mean_img = sum_img / count;
    double mean_templ = sum_templ / count;
    
    double numerator = sum_product - count * mean_img * mean_templ;
    double denom_img = sqrt(sum_img_sq - count * mean_img * mean_img);
    double denom_templ = sqrt(sum_templ_sq - count * mean_templ * mean_templ);
    
    if (denom_img * denom_templ < 1e-10) return 0.0;
    
    return numerator / (denom_img * denom_templ);
}

void find_best_match(unsigned char *image, unsigned char *templ,
                    int img_width, int img_height, int templ_size,
                    int *best_x, int *best_y, double *best_score) {
    *best_score = -1.0;
    
    for (int y = 0; y <= img_height - templ_size; y++) {
        for (int x = 0; x <= img_width - templ_size; x++) {
            double score = normalized_cross_correlation(image, templ, img_width, x, y, templ_size);
            
            if (score > *best_score) {
                *best_score = score;
                *best_x = x;
                *best_y = y;
            }
        }
    }
}

int main() {
    int width = IMAGE_WIDTH;
    int height = IMAGE_HEIGHT;
    int templ_size = TEMPLATE_SIZE;
    
    unsigned char *image = (unsigned char*)malloc(width * height);
    unsigned char *templ = (unsigned char*)malloc(templ_size * templ_size);
    
    unsigned int seed = 42;
    for (int i = 0; i < width * height; i++) {
        seed = seed * 1103515245 + 12345;
        image[i] = seed & 0xFF;
    }
    
    for (int i = 0; i < templ_size * templ_size; i++) {
        seed = seed * 1103515245 + 12345;
        templ[i] = seed & 0xFF;
    }
    
    clock_t start = clock();
    
    int best_x, best_y;
    double best_score;
    
    find_best_match(image, templ, width, height, templ_size, &best_x, &best_y, &best_score);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Template matching: %dx%d image, %dx%d template, %.6f seconds\n",
           width, height, templ_size, templ_size, time_spent);
    printf("Best match at (%d, %d) with score %.4f\n", best_x, best_y, best_score);
    
    free(image);
    free(templ);
    
    return 0;
}
