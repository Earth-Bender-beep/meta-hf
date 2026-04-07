#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int isSafe(int board[], int row, int col, int n) {
    for (int i = 0; i < row; i++) {
        if (board[i] == col || 
            board[i] - i == col - row || 
            board[i] + i == col + row)
            return 0;
    }
    return 1;
}

void solveNQueensUtil(int board[], int row, int n, int* count) {
    if (row == n) {
        (*count)++;
        return;
    }
    
    for (int col = 0; col < n; col++) {
        if (isSafe(board, row, col, n)) {
            board[row] = col;
            solveNQueensUtil(board, row + 1, n, count);
        }
    }
}

int solveNQueens(int n) {
    int* board = (int*)malloc(n * sizeof(int));
    int count = 0;
    
    solveNQueensUtil(board, 0, n, &count);
    
    free(board);
    return count;
}

int main() {
    int n = 13;
    
    clock_t start = clock();
    int solutions = solveNQueens(n);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("N-Queens: n=%d, solutions=%d in %.6f seconds\n", n, solutions, time_spent);
    
    return 0;
}
