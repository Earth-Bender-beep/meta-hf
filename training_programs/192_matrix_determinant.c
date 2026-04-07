// Matrix determinant computation using LU decomposition
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MATRIX_SIZE 50

double determinant_recursive(double **matrix, int n) {
    if (n == 1) return matrix[0][0];
    if (n == 2) return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    
    double det = 0;
    double **submatrix = (double**)malloc((n - 1) * sizeof(double*));
    for (int i = 0; i < n - 1; i++) {
        submatrix[i] = (double*)malloc((n - 1) * sizeof(double));
    }
    
    for (int x = 0; x < n; x++) {
        int subi = 0;
        for (int i = 1; i < n; i++) {
            int subj = 0;
            for (int j = 0; j < n; j++) {
                if (j == x) continue;
                submatrix[subi][subj] = matrix[i][j];
                subj++;
            }
            subi++;
        }
        
        det += (x % 2 == 0 ? 1 : -1) * matrix[0][x] * determinant_recursive(submatrix, n - 1);
    }
    
    for (int i = 0; i < n - 1; i++) {
        free(submatrix[i]);
    }
    free(submatrix);
    
    return det;
}

double determinant_lu(double **matrix, int n) {
    double **lu = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        lu[i] = (double*)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            lu[i][j] = matrix[i][j];
        }
    }
    
    double det = 1.0;
    
    for (int i = 0; i < n; i++) {
        int pivot = i;
        for (int j = i + 1; j < n; j++) {
            if (fabs(lu[j][i]) > fabs(lu[pivot][i])) {
                pivot = j;
            }
        }
        
        if (pivot != i) {
            double *temp = lu[i];
            lu[i] = lu[pivot];
            lu[pivot] = temp;
            det = -det;
        }
        
        if (fabs(lu[i][i]) < 1e-10) {
            for (int k = 0; k < n; k++) free(lu[k]);
            free(lu);
            return 0;
        }
        
        det *= lu[i][i];
        
        for (int j = i + 1; j < n; j++) {
            double factor = lu[j][i] / lu[i][i];
            for (int k = i; k < n; k++) {
                lu[j][k] -= factor * lu[i][k];
            }
        }
    }
    
    for (int i = 0; i < n; i++) free(lu[i]);
    free(lu);
    
    return det;
}

int main() {
    int n = 10;
    double **matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            seed = seed * 1103515245 + 12345;
            matrix[i][j] = ((seed & 0xFFFF) / (double)0xFFFF) * 10.0;
        }
    }
    
    clock_t start = clock();
    
    double det_lu = determinant_lu(matrix, n);
    
    for (int test = 0; test < 100; test++) {
        determinant_lu(matrix, n);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Matrix determinant: %dx%d matrix, 100 iterations, %.6f seconds\n",
           n, n, time_spent);
    printf("Determinant: %.6f\n", det_lu);
    
    for (int i = 0; i < n; i++) free(matrix[i]);
    free(matrix);
    
    return 0;
}
