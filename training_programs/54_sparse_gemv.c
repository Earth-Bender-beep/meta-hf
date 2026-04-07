// Sparse matrix-vector multiply using Compressed Sparse Row (CSR) format
// Memory-efficient for sparse matrices, different access pattern
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000
#define NNZ 5000  // Non-zero elements

typedef struct {
    int n;              // Matrix dimension
    int nnz;            // Number of non-zeros
    double *values;     // Non-zero values
    int *col_indices;   // Column indices
    int *row_ptr;       // Row pointers
} CSRMatrix;

CSRMatrix* create_csr_matrix(int n, int nnz) {
    CSRMatrix *mat = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    mat->n = n;
    mat->nnz = nnz;
    mat->values = (double*)malloc(nnz * sizeof(double));
    mat->col_indices = (int*)malloc(nnz * sizeof(int));
    mat->row_ptr = (int*)malloc((n + 1) * sizeof(int));
    return mat;
}

void free_csr_matrix(CSRMatrix *mat) {
    free(mat->values);
    free(mat->col_indices);
    free(mat->row_ptr);
    free(mat);
}

// SpMV: y = A*x
void sparse_gemv(CSRMatrix *A, double *x, double *y) {
    for (int i = 0; i < A->n; i++) {
        double sum = 0.0;
        for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
            sum += A->values[j] * x[A->col_indices[j]];
        }
        y[i] = sum;
    }
}

// Create test sparse matrix (tridiagonal + some random entries)
void init_sparse_matrix(CSRMatrix *mat) {
    int idx = 0;
    mat->row_ptr[0] = 0;
    
    for (int i = 0; i < mat->n; i++) {
        // Diagonal
        if (idx < mat->nnz) {
            mat->values[idx] = 4.0;
            mat->col_indices[idx] = i;
            idx++;
        }
        
        // Off-diagonal
        if (i > 0 && idx < mat->nnz) {
            mat->values[idx] = -1.0;
            mat->col_indices[idx] = i - 1;
            idx++;
        }
        
        if (i < mat->n - 1 && idx < mat->nnz) {
            mat->values[idx] = -1.0;
            mat->col_indices[idx] = i + 1;
            idx++;
        }
        
        // Add some random entries
        if (i % 10 == 0 && i + 5 < mat->n && idx < mat->nnz) {
            mat->values[idx] = 0.5;
            mat->col_indices[idx] = i + 5;
            idx++;
        }
        
        mat->row_ptr[i + 1] = idx;
    }
}

int main() {
    CSRMatrix *A = create_csr_matrix(N, NNZ);
    double *x = (double*)malloc(N * sizeof(double));
    double *y = (double*)malloc(N * sizeof(double));
    
    init_sparse_matrix(A);
    
    for (int i = 0; i < N; i++) {
        x[i] = (double)(i % 10) / 10.0;
    }
    
    clock_t start = clock();
    
    for (int iter = 0; iter < 1000; iter++) {
        sparse_gemv(A, x, y);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Sparse GEMV (CSR) %dx%d (nnz=%d, 1000 iters): %.6f seconds, y[0]=%.6f\n",
           N, N, A->nnz, time_spent, y[0]);
    
    free_csr_matrix(A);
    free(x);
    free(y);
    return 0;
}
