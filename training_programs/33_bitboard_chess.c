// Bitboard operations for chess-like games
// High bit manipulation, low memory ops - very different profile
#include <stdio.h>
#include <stdint.h>
#include <time.h>

typedef uint64_t Bitboard;

// Count bits (popcount)
int popcount(Bitboard bb) {
    int count = 0;
    while (bb) {
        count++;
        bb &= bb - 1; // Clear lowest set bit
    }
    return count;
}

// Get least significant bit
int lsb(Bitboard bb) {
    if (bb == 0) return -1;
    int pos = 0;
    while ((bb & 1) == 0) {
        bb >>= 1;
        pos++;
    }
    return pos;
}

// Knight move generation
Bitboard knight_attacks(int square) {
    Bitboard bb = 1ULL << square;
    Bitboard l1 = (bb >> 1) & 0x7f7f7f7f7f7f7f7fULL;
    Bitboard l2 = (bb >> 2) & 0x3f3f3f3f3f3f3f3fULL;
    Bitboard r1 = (bb << 1) & 0xfefefefefefefefeULL;
    Bitboard r2 = (bb << 2) & 0xfcfcfcfcfcfcfcfcULL;
    Bitboard h1 = l1 | r1;
    Bitboard h2 = l2 | r2;
    return (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8);
}

// Rook attacks (horizontal + vertical)
Bitboard rook_attacks(int square, Bitboard occupied) {
    Bitboard attacks = 0;
    int rank = square / 8;
    int file = square % 8;
    
    // North
    for (int r = rank + 1; r < 8; r++) {
        Bitboard sq = 1ULL << (r * 8 + file);
        attacks |= sq;
        if (occupied & sq) break;
    }
    
    // South
    for (int r = rank - 1; r >= 0; r--) {
        Bitboard sq = 1ULL << (r * 8 + file);
        attacks |= sq;
        if (occupied & sq) break;
    }
    
    // East
    for (int f = file + 1; f < 8; f++) {
        Bitboard sq = 1ULL << (rank * 8 + f);
        attacks |= sq;
        if (occupied & sq) break;
    }
    
    // West
    for (int f = file - 1; f >= 0; f--) {
        Bitboard sq = 1ULL << (rank * 8 + f);
        attacks |= sq;
        if (occupied & sq) break;
    }
    
    return attacks;
}

// Bishop attacks (diagonals)
Bitboard bishop_attacks(int square, Bitboard occupied) {
    Bitboard attacks = 0;
    int rank = square / 8;
    int file = square % 8;
    
    // NE
    for (int r = rank + 1, f = file + 1; r < 8 && f < 8; r++, f++) {
        Bitboard sq = 1ULL << (r * 8 + f);
        attacks |= sq;
        if (occupied & sq) break;
    }
    
    // SE
    for (int r = rank - 1, f = file + 1; r >= 0 && f < 8; r--, f++) {
        Bitboard sq = 1ULL << (r * 8 + f);
        attacks |= sq;
        if (occupied & sq) break;
    }
    
    // SW
    for (int r = rank - 1, f = file - 1; r >= 0 && f >= 0; r--, f--) {
        Bitboard sq = 1ULL << (r * 8 + f);
        attacks |= sq;
        if (occupied & sq) break;
    }
    
    // NW
    for (int r = rank + 1, f = file - 1; r < 8 && f >= 0; r++, f--) {
        Bitboard sq = 1ULL << (r * 8 + f);
        attacks |= sq;
        if (occupied & sq) break;
    }
    
    return attacks;
}

int main() {
    Bitboard occupied = 0x0000001008100000ULL; // Some pieces
    int total_attacks = 0;
    
    clock_t start = clock();
    
    // Compute attacks for all squares
    for (int sq = 0; sq < 64; sq++) {
        Bitboard knight = knight_attacks(sq);
        Bitboard rook = rook_attacks(sq, occupied);
        Bitboard bishop = bishop_attacks(sq, occupied);
        
        total_attacks += popcount(knight);
        total_attacks += popcount(rook);
        total_attacks += popcount(bishop);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Bitboard chess attacks: %.6f seconds, total=%d\n", 
           time_spent, total_attacks);
    
    return 0;
}
