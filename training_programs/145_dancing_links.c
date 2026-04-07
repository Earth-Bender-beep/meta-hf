// Dancing Links (DLX) - Knuth's Algorithm X for exact cover problems
// Efficient backtracking using doubly-linked matrix structure
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_NODES 10000
#define MAX_COLS 100
#define NUM_SUDOKU_RUNS 10

typedef struct Node {
    struct Node *left, *right, *up, *down;
    struct ColumnNode *column;
    int row_id;
} Node;

typedef struct ColumnNode {
    Node header;
    int size;
    int id;
} ColumnNode;

typedef struct {
    ColumnNode *head;
    ColumnNode columns[MAX_COLS];
    Node nodes[MAX_NODES];
    int node_count;
    int *solution;
    int solution_size;
    int solutions_found;
} DLXMatrix;

void init_column(ColumnNode *col, int id) {
    col->size = 0;
    col->id = id;
    col->header.left = &col->header;
    col->header.right = &col->header;
    col->header.up = &col->header;
    col->header.down = &col->header;
    col->header.column = col;
}

DLXMatrix* create_dlx_matrix(int num_cols) {
    DLXMatrix *dlx = (DLXMatrix*)malloc(sizeof(DLXMatrix));
    dlx->node_count = 0;
    dlx->solution_size = 0;
    dlx->solutions_found = 0;
    dlx->solution = (int*)malloc(1000 * sizeof(int));
    
    // Initialize head
    dlx->head = &dlx->columns[0];
    init_column(dlx->head, -1);
    
    // Initialize columns
    for (int i = 1; i <= num_cols; i++) {
        init_column(&dlx->columns[i], i);
        
        // Link columns horizontally
        dlx->columns[i].header.left = dlx->head->header.left;
        dlx->columns[i].header.right = &dlx->head->header;
        dlx->head->header.left->right = &dlx->columns[i].header;
        dlx->head->header.left = &dlx->columns[i].header;
    }
    
    return dlx;
}

void add_row(DLXMatrix *dlx, int row_id, int *cols, int num_cols_in_row) {
    Node *first = NULL;
    Node *last = NULL;
    
    for (int i = 0; i < num_cols_in_row; i++) {
        if (dlx->node_count >= MAX_NODES) return;
        
        Node *node = &dlx->nodes[dlx->node_count++];
        ColumnNode *col = &dlx->columns[cols[i]];
        
        node->row_id = row_id;
        node->column = col;
        
        // Link vertically
        node->up = col->header.up;
        node->down = &col->header;
        col->header.up->down = node;
        col->header.up = node;
        col->size++;
        
        // Link horizontally
        if (first == NULL) {
            first = node;
            node->left = node;
            node->right = node;
        } else {
            node->left = last;
            node->right = first;
            last->right = node;
            first->left = node;
        }
        last = node;
    }
}

void cover_column(ColumnNode *col) {
    col->header.right->left = col->header.left;
    col->header.left->right = col->header.right;
    
    for (Node *row = col->header.down; row != &col->header; row = row->down) {
        for (Node *node = row->right; node != row; node = node->right) {
            node->down->up = node->up;
            node->up->down = node->down;
            node->column->size--;
        }
    }
}

void uncover_column(ColumnNode *col) {
    for (Node *row = col->header.up; row != &col->header; row = row->up) {
        for (Node *node = row->left; node != row; node = node->left) {
            node->column->size++;
            node->down->up = node;
            node->up->down = node;
        }
    }
    
    col->header.right->left = &col->header;
    col->header.left->right = &col->header;
}

ColumnNode* choose_column(DLXMatrix *dlx) {
    ColumnNode *best = NULL;
    int min_size = MAX_NODES + 1;
    
    for (Node *node = dlx->head->header.right; 
         node != &dlx->head->header; 
         node = node->right) {
        ColumnNode *col = (ColumnNode*)node;
        if (col->size < min_size) {
            min_size = col->size;
            best = col;
        }
    }
    
    return best;
}

int search(DLXMatrix *dlx, int max_solutions) {
    if (dlx->head->header.right == &dlx->head->header) {
        dlx->solutions_found++;
        return 1;
    }
    
    if (dlx->solutions_found >= max_solutions) {
        return 1;
    }
    
    ColumnNode *col = choose_column(dlx);
    if (col == NULL || col->size == 0) return 0;
    
    cover_column(col);
    
    for (Node *row = col->header.down; row != &col->header; row = row->down) {
        dlx->solution[dlx->solution_size++] = row->row_id;
        
        for (Node *node = row->right; node != row; node = node->right) {
            cover_column(node->column);
        }
        
        if (search(dlx, max_solutions)) {
            if (dlx->solutions_found >= max_solutions) {
                return 1;
            }
        }
        
        dlx->solution_size--;
        
        for (Node *node = row->left; node != row; node = node->left) {
            uncover_column(node->column);
        }
    }
    
    uncover_column(col);
    return 0;
}

// Create a simple exact cover problem
void create_test_problem(DLXMatrix *dlx, unsigned int *seed) {
    int num_cols = 20;
    int num_rows = 30;
    
    for (int r = 0; r < num_rows; r++) {
        int cols[5];
        int count = 0;
        
        for (int c = 0; c < 5; c++) {
            *seed = *seed * 1103515245 + 12345;
            int col_id = (*seed % num_cols) + 1;
            cols[count++] = col_id;
        }
        
        add_row(dlx, r, cols, count);
    }
}

int main() {
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    int total_solutions = 0;
    
    for (int run = 0; run < NUM_SUDOKU_RUNS; run++) {
        DLXMatrix *dlx = create_dlx_matrix(20);
        create_test_problem(dlx, &seed);
        
        search(dlx, 5);  // Find up to 5 solutions
        total_solutions += dlx->solutions_found;
        
        free(dlx->solution);
        free(dlx);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Dancing Links (Algorithm X): %d problems, %.6f seconds\n",
           NUM_SUDOKU_RUNS, time_spent);
    printf("Total solutions found: %d\n", total_solutions);
    
    return 0;
}
