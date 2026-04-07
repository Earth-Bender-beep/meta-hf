// Minimax algorithm for Tic-Tac-Toe game tree search
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BOARD_SIZE 3
#define EMPTY 0
#define PLAYER_X 1
#define PLAYER_O 2

typedef struct {
    int board[BOARD_SIZE][BOARD_SIZE];
} Board;

int check_winner(Board *b) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (b->board[i][0] == b->board[i][1] && b->board[i][1] == b->board[i][2] && b->board[i][0] != EMPTY) {
            return b->board[i][0];
        }
        if (b->board[0][i] == b->board[1][i] && b->board[1][i] == b->board[2][i] && b->board[0][i] != EMPTY) {
            return b->board[0][i];
        }
    }
    
    if (b->board[0][0] == b->board[1][1] && b->board[1][1] == b->board[2][2] && b->board[0][0] != EMPTY) {
        return b->board[0][0];
    }
    if (b->board[0][2] == b->board[1][1] && b->board[1][1] == b->board[2][0] && b->board[0][2] != EMPTY) {
        return b->board[0][2];
    }
    
    return 0;
}

int is_full(Board *b) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (b->board[i][j] == EMPTY) return 0;
        }
    }
    return 1;
}

int minimax(Board *b, int depth, int is_maximizing) {
    int winner = check_winner(b);
    
    if (winner == PLAYER_X) return 10 - depth;
    if (winner == PLAYER_O) return depth - 10;
    if (is_full(b)) return 0;
    
    if (is_maximizing) {
        int best_score = -1000;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (b->board[i][j] == EMPTY) {
                    b->board[i][j] = PLAYER_X;
                    int score = minimax(b, depth + 1, 0);
                    b->board[i][j] = EMPTY;
                    if (score > best_score) best_score = score;
                }
            }
        }
        return best_score;
    } else {
        int best_score = 1000;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (b->board[i][j] == EMPTY) {
                    b->board[i][j] = PLAYER_O;
                    int score = minimax(b, depth + 1, 1);
                    b->board[i][j] = EMPTY;
                    if (score < best_score) best_score = score;
                }
            }
        }
        return best_score;
    }
}

void find_best_move(Board *b, int *best_x, int *best_y) {
    int best_score = -1000;
    
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (b->board[i][j] == EMPTY) {
                b->board[i][j] = PLAYER_X;
                int score = minimax(b, 0, 0);
                b->board[i][j] = EMPTY;
                
                if (score > best_score) {
                    best_score = score;
                    *best_x = i;
                    *best_y = j;
                }
            }
        }
    }
}

int main() {
    Board board;
    
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            board.board[i][j] = EMPTY;
        }
    }
    
    clock_t start = clock();
    
    int games_played = 0;
    
    for (int test = 0; test < 100; test++) {
        Board test_board;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                test_board.board[i][j] = EMPTY;
            }
        }
        
        int best_x, best_y;
        find_best_move(&test_board, &best_x, &best_y);
        
        test_board.board[best_x][best_y] = PLAYER_X;
        games_played++;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Minimax Tic-Tac-Toe: %d games, %.6f seconds\n", games_played, time_spent);
    
    return 0;
}
