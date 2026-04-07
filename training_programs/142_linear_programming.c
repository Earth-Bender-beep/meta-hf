// Simplex algorithm for linear programming optimization
// Solve linear optimization problems with constraints
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

#define MAX_VARS 10
#define MAX_CONSTRAINTS 10
#define EPSILON 1e-6

typedef struct {
    double tableau[MAX_CONSTRAINTS + 1][MAX_VARS + MAX_CONSTRAINTS + 1];
    int num_vars;
    int num_constraints;
    int num_cols;
} SimplexTableau;

void initialize_tableau(SimplexTableau *st, double obj[], double constraints[][MAX_VARS + 1], 
                        int num_vars, int num_constraints) {
    st->num_vars = num_vars;
    st->num_constraints = num_constraints;
    st->num_cols = num_vars + num_constraints + 1;
    
    // Initialize tableau with zeros
    for (int i = 0; i <= num_constraints; i++) {
        for (int j = 0; j < st->num_cols; j++) {
            st->tableau[i][j] = 0.0;
        }
    }
    
    // Set objective function (negated for maximization)
    for (int j = 0; j < num_vars; j++) {
        st->tableau[0][j] = -obj[j];
    }
    
    // Set constraints
    for (int i = 0; i < num_constraints; i++) {
        for (int j = 0; j < num_vars; j++) {
            st->tableau[i + 1][j] = constraints[i][j];
        }
        // Slack variable
        st->tableau[i + 1][num_vars + i] = 1.0;
        // RHS
        st->tableau[i + 1][st->num_cols - 1] = constraints[i][num_vars];
    }
}

int find_pivot_column(SimplexTableau *st) {
    int pivot_col = -1;
    double min_value = 0.0;
    
    for (int j = 0; j < st->num_cols - 1; j++) {
        if (st->tableau[0][j] < min_value) {
            min_value = st->tableau[0][j];
            pivot_col = j;
        }
    }
    
    return pivot_col;
}

int find_pivot_row(SimplexTableau *st, int pivot_col) {
    int pivot_row = -1;
    double min_ratio = DBL_MAX;
    
    for (int i = 1; i <= st->num_constraints; i++) {
        if (st->tableau[i][pivot_col] > EPSILON) {
            double ratio = st->tableau[i][st->num_cols - 1] / st->tableau[i][pivot_col];
            if (ratio >= 0 && ratio < min_ratio) {
                min_ratio = ratio;
                pivot_row = i;
            }
        }
    }
    
    return pivot_row;
}

void pivot(SimplexTableau *st, int pivot_row, int pivot_col) {
    double pivot_value = st->tableau[pivot_row][pivot_col];
    
    // Normalize pivot row
    for (int j = 0; j < st->num_cols; j++) {
        st->tableau[pivot_row][j] /= pivot_value;
    }
    
    // Eliminate other rows
    for (int i = 0; i <= st->num_constraints; i++) {
        if (i != pivot_row) {
            double factor = st->tableau[i][pivot_col];
            for (int j = 0; j < st->num_cols; j++) {
                st->tableau[i][j] -= factor * st->tableau[pivot_row][j];
            }
        }
    }
}

double simplex(SimplexTableau *st) {
    int iterations = 0;
    int max_iterations = 100;
    
    while (iterations < max_iterations) {
        int pivot_col = find_pivot_column(st);
        
        if (pivot_col == -1) {
            // Optimal solution found
            break;
        }
        
        int pivot_row = find_pivot_row(st, pivot_col);
        
        if (pivot_row == -1) {
            // Unbounded problem
            return DBL_MAX;
        }
        
        pivot(st, pivot_row, pivot_col);
        iterations++;
    }
    
    return st->tableau[0][st->num_cols - 1];
}

int main() {
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    double total_optimal = 0.0;
    int num_problems = 100;
    
    for (int prob = 0; prob < num_problems; prob++) {
        // Generate random linear programming problem
        int num_vars = 3 + (seed % 5);
        int num_constraints = 3 + (seed % 5);
        seed = seed * 1103515245 + 12345;
        
        double obj[MAX_VARS];
        double constraints[MAX_CONSTRAINTS][MAX_VARS + 1];
        
        // Random objective function
        for (int j = 0; j < num_vars; j++) {
            seed = seed * 1103515245 + 12345;
            obj[j] = (seed % 100) / 10.0;
        }
        
        // Random constraints
        for (int i = 0; i < num_constraints; i++) {
            for (int j = 0; j < num_vars; j++) {
                seed = seed * 1103515245 + 12345;
                constraints[i][j] = (seed % 50) / 10.0;
            }
            seed = seed * 1103515245 + 12345;
            constraints[i][num_vars] = 50.0 + (seed % 100);
        }
        
        SimplexTableau st;
        initialize_tableau(&st, obj, constraints, num_vars, num_constraints);
        
        double optimal_value = simplex(&st);
        
        if (optimal_value != DBL_MAX) {
            total_optimal += optimal_value;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Simplex algorithm: %d LP problems, %.6f seconds\n", num_problems, time_spent);
    printf("Total optimal value: %.2f\n", total_optimal);
    
    return 0;
}
