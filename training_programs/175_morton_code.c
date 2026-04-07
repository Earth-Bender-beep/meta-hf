// Morton code (Z-order curve) for spatial indexing
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_POINTS 100000

unsigned int part1by1(unsigned int n) {
    n &= 0x0000FFFF;
    n = (n | (n << 8)) & 0x00FF00FF;
    n = (n | (n << 4)) & 0x0F0F0F0F;
    n = (n | (n << 2)) & 0x33333333;
    n = (n | (n << 1)) & 0x55555555;
    return n;
}

unsigned int unpart1by1(unsigned int n) {
    n &= 0x55555555;
    n = (n | (n >> 1)) & 0x33333333;
    n = (n | (n >> 2)) & 0x0F0F0F0F;
    n = (n | (n >> 4)) & 0x00FF00FF;
    n = (n | (n >> 8)) & 0x0000FFFF;
    return n;
}

unsigned int part1by2(unsigned int n) {
    n &= 0x000003FF;
    n = (n | (n << 16)) & 0xFF0000FF;
    n = (n | (n << 8)) & 0x0300F00F;
    n = (n | (n << 4)) & 0x030C30C3;
    n = (n | (n << 2)) & 0x09249249;
    return n;
}

unsigned int unpart1by2(unsigned int n) {
    n &= 0x09249249;
    n = (n | (n >> 2)) & 0x030C30C3;
    n = (n | (n >> 4)) & 0x0300F00F;
    n = (n | (n >> 8)) & 0xFF0000FF;
    n = (n | (n >> 16)) & 0x000003FF;
    return n;
}

unsigned int encode_morton2(unsigned int x, unsigned int y) {
    return part1by1(x) | (part1by1(y) << 1);
}

void decode_morton2(unsigned int code, unsigned int *x, unsigned int *y) {
    *x = unpart1by1(code);
    *y = unpart1by1(code >> 1);
}

unsigned int encode_morton3(unsigned int x, unsigned int y, unsigned int z) {
    return part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2);
}

void decode_morton3(unsigned int code, unsigned int *x, unsigned int *y, unsigned int *z) {
    *x = unpart1by2(code);
    *y = unpart1by2(code >> 1);
    *z = unpart1by2(code >> 2);
}

int main() {
    unsigned int *morton_codes_2d = (unsigned int*)malloc(NUM_POINTS * sizeof(unsigned int));
    unsigned int *morton_codes_3d = (unsigned int*)malloc(NUM_POINTS * sizeof(unsigned int));
    
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    for (int i = 0; i < NUM_POINTS; i++) {
        seed = seed * 1103515245 + 12345;
        unsigned int x = (seed & 0xFFFF) % 1024;
        seed = seed * 1103515245 + 12345;
        unsigned int y = (seed & 0xFFFF) % 1024;
        
        morton_codes_2d[i] = encode_morton2(x, y);
    }
    
    for (int i = 0; i < NUM_POINTS; i++) {
        seed = seed * 1103515245 + 12345;
        unsigned int x = (seed & 0x3FF);
        seed = seed * 1103515245 + 12345;
        unsigned int y = (seed & 0x3FF);
        seed = seed * 1103515245 + 12345;
        unsigned int z = (seed & 0x3FF);
        
        morton_codes_3d[i] = encode_morton3(x, y, z);
    }
    
    long long sum_2d = 0;
    for (int i = 0; i < NUM_POINTS; i++) {
        unsigned int x, y;
        decode_morton2(morton_codes_2d[i], &x, &y);
        sum_2d += x + y;
    }
    
    long long sum_3d = 0;
    for (int i = 0; i < NUM_POINTS; i++) {
        unsigned int x, y, z;
        decode_morton3(morton_codes_3d[i], &x, &y, &z);
        sum_3d += x + y + z;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Morton code: %d points, %.6f seconds\n", NUM_POINTS, time_spent);
    printf("Sum 2D: %lld, Sum 3D: %lld\n", sum_2d, sum_3d);
    
    free(morton_codes_2d);
    free(morton_codes_3d);
    
    return 0;
}
