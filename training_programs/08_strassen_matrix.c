#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void add_matrix(int** A, int** B, int** C, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            C[i][j] = A[i][j] + B[i][j];
}

void sub_matrix(int** A, int** B, int** C, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            C[i][j] = A[i][j] - B[i][j];
}

void simple_multiply(int** A, int** B, int** C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = 0;
            for (int k = 0; k < size; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
    }
}

int main() {
    int n = 256;
    
    int** A = (int**)malloc(n * sizeof(int*));
    int** B = (int**)malloc(n * sizeof(int*));
    int** C = (int**)malloc(n * sizeof(int*));
    
    for (int i = 0; i < n; i++) {
        A[i] = (int*)malloc(n * sizeof(int));
        B[i] = (int*)malloc(n * sizeof(int));
        C[i] = (int*)malloc(n * sizeof(int));
    }
    
    srand(42);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }
    
    clock_t start = clock();
    simple_multiply(A, B, C, n);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Matrix multiply (%dx%d): %.6f seconds\n", n, n, time_spent);
    
    for (int i = 0; i < n; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);
    
    return 0;
}
