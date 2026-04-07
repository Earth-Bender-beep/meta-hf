// Sparse matrix operations using Compressed Sparse Row (CSR) format
// Efficient storage and operations for matrices with many zeros
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_SIZE 1000
#define SPARSITY 0.95  // 95% zeros
#define NUM_OPERATIONS 1000

typedef struct {
    double *values;
    int *col_indices;
    int *row_ptr;
    int num_rows;
    int num_nonzero;
} SparseMatrixCSR;

SparseMatrixCSR* create_sparse_matrix(int rows, int nnz) {
    SparseMatrixCSR *mat = (SparseMatrixCSR*)malloc(sizeof(SparseMatrixCSR));
    mat->values = (double*)malloc(nnz * sizeof(double));
    mat->col_indices = (int*)malloc(nnz * sizeof(int));
    mat->row_ptr = (int*)malloc((rows + 1) * sizeof(int));
    mat->num_rows = rows;
    mat->num_nonzero = nnz;
    return mat;
}

void generate_sparse_matrix(SparseMatrixCSR *mat, int size) {
    unsigned int seed = 42;
    int nnz_per_row = (int)((1.0 - SPARSITY) * size);
    
    mat->row_ptr[0] = 0;
    int idx = 0;
    
    for (int i = 0; i < size; i++) {
        // Generate random non-zero elements for this row
        for (int j = 0; j < nnz_per_row && idx < mat->num_nonzero; j++) {
            seed = seed * 1103515245 + 12345;
            int col = (seed % size);
            seed = seed * 1103515245 + 12345;
            double val = ((seed & 0xFFFF) / (double)0xFFFF) * 10.0;
            
            mat->values[idx] = val;
            mat->col_indices[idx] = col;
            idx++;
        }
        mat->row_ptr[i + 1] = idx;
    }
}

void sparse_matrix_vector_multiply(SparseMatrixCSR *mat, double *vec, double *result) {
    for (int i = 0; i < mat->num_rows; i++) {
        result[i] = 0.0;
        for (int j = mat->row_ptr[i]; j < mat->row_ptr[i + 1]; j++) {
            result[i] += mat->values[j] * vec[mat->col_indices[j]];
        }
    }
}

void sparse_matrix_transpose(SparseMatrixCSR *mat, SparseMatrixCSR *result) {
    // Count elements in each column
    int *col_count = (int*)calloc(mat->num_rows, sizeof(int));
    
    for (int i = 0; i < mat->num_nonzero; i++) {
        col_count[mat->col_indices[i]]++;
    }
    
    // Set up row pointers for result
    result->row_ptr[0] = 0;
    for (int i = 0; i < mat->num_rows; i++) {
        result->row_ptr[i + 1] = result->row_ptr[i] + col_count[i];
    }
    
    // Fill in values and column indices
    int *curr_pos = (int*)calloc(mat->num_rows, sizeof(int));
    for (int i = 0; i < mat->num_rows; i++) {
        curr_pos[i] = result->row_ptr[i];
    }
    
    for (int i = 0; i < mat->num_rows; i++) {
        for (int j = mat->row_ptr[i]; j < mat->row_ptr[i + 1]; j++) {
            int col = mat->col_indices[j];
            int pos = curr_pos[col]++;
            result->values[pos] = mat->values[j];
            result->col_indices[pos] = i;
        }
    }
    
    free(col_count);
    free(curr_pos);
}

double sparse_matrix_norm(SparseMatrixCSR *mat) {
    double sum = 0.0;
    for (int i = 0; i < mat->num_nonzero; i++) {
        sum += mat->values[i] * mat->values[i];
    }
    return sum;
}

int main() {
    int nnz = (int)(MATRIX_SIZE * MATRIX_SIZE * (1.0 - SPARSITY));
    
    SparseMatrixCSR *mat = create_sparse_matrix(MATRIX_SIZE, nnz);
    generate_sparse_matrix(mat, MATRIX_SIZE);
    
    double *vec = (double*)malloc(MATRIX_SIZE * sizeof(double));
    double *result = (double*)malloc(MATRIX_SIZE * sizeof(double));
    
    unsigned int seed = 123;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        seed = seed * 1103515245 + 12345;
        vec[i] = ((seed & 0xFFFF) / (double)0xFFFF);
    }
    
    clock_t start = clock();
    
    // Matrix-vector multiplications
    for (int i = 0; i < NUM_OPERATIONS; i++) {
        sparse_matrix_vector_multiply(mat, vec, result);
    }
    
    // Transpose operation
    SparseMatrixCSR *mat_T = create_sparse_matrix(MATRIX_SIZE, nnz);
    sparse_matrix_transpose(mat, mat_T);
    
    double norm = sparse_matrix_norm(mat);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Sparse matrix (CSR): size=%dx%d, sparsity=%.1f%%, nnz=%d\n",
           MATRIX_SIZE, MATRIX_SIZE, SPARSITY * 100, nnz);
    printf("%d operations, %.6f seconds\n", NUM_OPERATIONS, time_spent);
    printf("Matrix norm: %.4f\n", norm);
    
    free(mat->values);
    free(mat->col_indices);
    free(mat->row_ptr);
    free(mat);
    free(mat_T->values);
    free(mat_T->col_indices);
    free(mat_T->row_ptr);
    free(mat_T);
    free(vec);
    free(result);
    
    return 0;
}
