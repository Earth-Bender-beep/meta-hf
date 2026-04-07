// Discrete Cosine Transform (DCT) Type-II for signal compression
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 8
#define NUM_BLOCKS 1000

void dct_1d(double *input, double *output, int n) {
    for (int k = 0; k < n; k++) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += input[i] * cos(M_PI * k * (2 * i + 1) / (2.0 * n));
        }
        
        double scale = (k == 0) ? sqrt(1.0 / n) : sqrt(2.0 / n);
        output[k] = scale * sum;
    }
}

void idct_1d(double *input, double *output, int n) {
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            double scale = (k == 0) ? sqrt(1.0 / n) : sqrt(2.0 / n);
            sum += scale * input[k] * cos(M_PI * k * (2 * i + 1) / (2.0 * n));
        }
        output[i] = sum;
    }
}

void dct_2d(double block[BLOCK_SIZE][BLOCK_SIZE], double output[BLOCK_SIZE][BLOCK_SIZE]) {
    double temp[BLOCK_SIZE][BLOCK_SIZE];
    
    // DCT on rows
    for (int i = 0; i < BLOCK_SIZE; i++) {
        dct_1d(block[i], temp[i], BLOCK_SIZE);
    }
    
    // DCT on columns
    double col_in[BLOCK_SIZE], col_out[BLOCK_SIZE];
    for (int j = 0; j < BLOCK_SIZE; j++) {
        for (int i = 0; i < BLOCK_SIZE; i++) {
            col_in[i] = temp[i][j];
        }
        dct_1d(col_in, col_out, BLOCK_SIZE);
        for (int i = 0; i < BLOCK_SIZE; i++) {
            output[i][j] = col_out[i];
        }
    }
}

void quantize(double block[BLOCK_SIZE][BLOCK_SIZE], int quant_table[BLOCK_SIZE][BLOCK_SIZE]) {
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            block[i][j] = round(block[i][j] / quant_table[i][j]);
        }
    }
}

int main() {
    double (*blocks)[BLOCK_SIZE][BLOCK_SIZE] = malloc(NUM_BLOCKS * sizeof(*blocks));
    double (*dct_blocks)[BLOCK_SIZE][BLOCK_SIZE] = malloc(NUM_BLOCKS * sizeof(*dct_blocks));
    
    int quant_table[BLOCK_SIZE][BLOCK_SIZE] = {
        {16, 11, 10, 16, 24, 40, 51, 61},
        {12, 12, 14, 19, 26, 58, 60, 55},
        {14, 13, 16, 24, 40, 57, 69, 56},
        {14, 17, 22, 29, 51, 87, 80, 62},
        {18, 22, 37, 56, 68, 109, 103, 77},
        {24, 35, 55, 64, 81, 104, 113, 92},
        {49, 64, 78, 87, 103, 121, 120, 101},
        {72, 92, 95, 98, 112, 100, 103, 99}
    };
    
    unsigned int seed = 42;
    for (int b = 0; b < NUM_BLOCKS; b++) {
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                seed = seed * 1103515245 + 12345;
                blocks[b][i][j] = ((seed & 0xFF) / (double)0xFF) * 255.0 - 128.0;
            }
        }
    }
    
    clock_t start = clock();
    
    for (int b = 0; b < NUM_BLOCKS; b++) {
        dct_2d(blocks[b], dct_blocks[b]);
        quantize(dct_blocks[b], quant_table);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("DCT Transform: %d blocks of %dx%d, %.6f seconds\n",
           NUM_BLOCKS, BLOCK_SIZE, BLOCK_SIZE, time_spent);
    
    free(blocks);
    free(dct_blocks);
    
    return 0;
}
