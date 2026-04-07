// Median filter for image noise reduction
// Sorting-based, sliding window, different from other filters
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 256
#define HEIGHT 256
#define WINDOW_SIZE 5

typedef struct {
    unsigned char data[HEIGHT][WIDTH];
} Image;

void insertion_sort(unsigned char *arr, int n) {
    for (int i = 1; i < n; i++) {
        unsigned char key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

unsigned char find_median(unsigned char *values, int count) {
    insertion_sort(values, count);
    return values[count / 2];
}

void median_filter(Image *input, Image *output, int window_size) {
    int half = window_size / 2;
    int max_values = window_size * window_size;
    unsigned char *window = (unsigned char*)malloc(max_values);
    
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int count = 0;
            
            for (int wy = -half; wy <= half; wy++) {
                for (int wx = -half; wx <= half; wx++) {
                    int py = y + wy;
                    int px = x + wx;
                    
                    if (py >= 0 && py < HEIGHT && px >= 0 && px < WIDTH) {
                        window[count++] = input->data[py][px];
                    }
                }
            }
            
            output->data[y][x] = find_median(window, count);
        }
    }
    
    free(window);
}

void add_salt_pepper_noise(Image *img, int noise_level) {
    for (int i = 0; i < noise_level; i++) {
        int x = rand() % WIDTH;
        int y = rand() % HEIGHT;
        img->data[y][x] = (rand() % 2) ? 255 : 0;
    }
}

void init_image(Image *img) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            img->data[y][x] = (x + y * 2) % 256;
        }
    }
}

int main() {
    Image *input = (Image*)malloc(sizeof(Image));
    Image *output = (Image*)malloc(sizeof(Image));
    
    srand(42);
    init_image(input);
    add_salt_pepper_noise(input, 1000);
    
    clock_t start = clock();
    median_filter(input, output, WINDOW_SIZE);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Median filter %dx%d (window=%d): %.6f seconds, pixel[128][128]=%d\n",
           WIDTH, HEIGHT, WINDOW_SIZE, time_spent, output->data[128][128]);
    
    free(input);
    free(output);
    return 0;
}
