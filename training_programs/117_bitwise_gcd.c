// Binary GCD algorithm (Stein's algorithm)
// GCD computation using bit operations instead of division
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_TESTS 1000000

unsigned long long binary_gcd(unsigned long long u, unsigned long long v) {
    if (u == 0) return v;
    if (v == 0) return u;
    
    // Count common factors of 2
    int shift = 0;
    while (((u | v) & 1) == 0) {
        u >>= 1;
        v >>= 1;
        shift++;
    }
    
    // Remove remaining factors of 2 from u
    while ((u & 1) == 0) {
        u >>= 1;
    }
    
    do {
        // Remove factors of 2 from v
        while ((v & 1) == 0) {
            v >>= 1;
        }
        
        // Ensure u <= v
        if (u > v) {
            unsigned long long temp = u;
            u = v;
            v = temp;
        }
        
        v -= u;
    } while (v != 0);
    
    return u << shift;
}

// Regular Euclidean GCD for comparison
unsigned long long euclidean_gcd(unsigned long long a, unsigned long long b) {
    while (b != 0) {
        unsigned long long temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

int main() {
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    unsigned long long sum = 0;
    for (int i = 0; i < NUM_TESTS; i++) {
        seed = seed * 1103515245 + 12345;
        unsigned long long a = (seed & 0xFFFFFFF) + 1;
        seed = seed * 1103515245 + 12345;
        unsigned long long b = (seed & 0xFFFFFFF) + 1;
        
        sum += binary_gcd(a, b);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Binary GCD (Stein's): %d tests, %.6f seconds\n",
           NUM_TESTS, time_spent);
    printf("Sum of GCDs: %llu\n", sum);
    
    return 0;
}
