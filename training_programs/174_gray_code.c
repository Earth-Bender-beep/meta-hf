// Gray code generation and conversion algorithms
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_BITS 20
#define NUM_TESTS 500000

unsigned int binary_to_gray(unsigned int n) {
    return n ^ (n >> 1);
}

unsigned int gray_to_binary(unsigned int gray) {
    unsigned int binary = gray;
    while (gray >>= 1) {
        binary ^= gray;
    }
    return binary;
}

void generate_gray_sequence(unsigned int *sequence, int n) {
    for (int i = 0; i < (1 << n); i++) {
        sequence[i] = binary_to_gray(i);
    }
}

int hamming_distance(unsigned int a, unsigned int b) {
    unsigned int xor_result = a ^ b;
    int count = 0;
    while (xor_result) {
        count += xor_result & 1;
        xor_result >>= 1;
    }
    return count;
}

int verify_gray_property(unsigned int *sequence, int length) {
    for (int i = 0; i < length - 1; i++) {
        if (hamming_distance(sequence[i], sequence[i + 1]) != 1) {
            return 0;
        }
    }
    return 1;
}

void next_gray_permutation(unsigned int *gray, int n) {
    unsigned int current = *gray;
    unsigned int binary = gray_to_binary(current);
    binary++;
    *gray = binary_to_gray(binary);
}

int main() {
    int n = 16;
    int size = 1 << n;
    unsigned int *sequence = (unsigned int*)malloc(size * sizeof(unsigned int));
    
    clock_t start = clock();
    
    generate_gray_sequence(sequence, n);
    
    int is_valid = verify_gray_property(sequence, size);
    
    long long conversion_sum = 0;
    for (int i = 0; i < NUM_TESTS; i++) {
        unsigned int gray = binary_to_gray(i);
        unsigned int binary = gray_to_binary(gray);
        conversion_sum += binary;
    }
    
    long long distance_sum = 0;
    for (int i = 0; i < 10000; i++) {
        distance_sum += hamming_distance(sequence[i], sequence[i + 1]);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Gray code: %d bits, %d codes generated, %.6f seconds\n",
           n, size, time_spent);
    printf("Gray property valid: %s\n", is_valid ? "YES" : "NO");
    printf("Conversion sum: %lld, Distance sum: %lld\n", conversion_sum, distance_sum);
    
    free(sequence);
    
    return 0;
}
