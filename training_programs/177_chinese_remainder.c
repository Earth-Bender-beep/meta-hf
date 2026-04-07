// Chinese Remainder Theorem for modular arithmetic
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_EQUATIONS 10

long long extended_gcd(long long a, long long b, long long *x, long long *y) {
    if (b == 0) {
        *x = 1;
        *y = 0;
        return a;
    }
    
    long long x1, y1;
    long long gcd = extended_gcd(b, a % b, &x1, &y1);
    
    *x = y1;
    *y = x1 - (a / b) * y1;
    
    return gcd;
}

long long mod_inverse(long long a, long long m) {
    long long x, y;
    long long gcd = extended_gcd(a, m, &x, &y);
    
    if (gcd != 1) return -1;
    
    return (x % m + m) % m;
}

long long chinese_remainder_theorem(long long *remainders, long long *moduli, int n) {
    long long product = 1;
    for (int i = 0; i < n; i++) {
        product *= moduli[i];
    }
    
    long long result = 0;
    
    for (int i = 0; i < n; i++) {
        long long partial_product = product / moduli[i];
        long long inverse = mod_inverse(partial_product, moduli[i]);
        result += remainders[i] * partial_product * inverse;
    }
    
    return result % product;
}

int main() {
    long long remainders[NUM_EQUATIONS];
    long long moduli[NUM_EQUATIONS];
    
    unsigned int seed = 42;
    
    for (int i = 0; i < NUM_EQUATIONS; i++) {
        seed = seed * 1103515245 + 12345;
        moduli[i] = (seed % 10) + 11;
        seed = seed * 1103515245 + 12345;
        remainders[i] = seed % moduli[i];
    }
    
    clock_t start = clock();
    
    long long results[100];
    for (int test = 0; test < 100; test++) {
        results[test] = chinese_remainder_theorem(remainders, moduli, NUM_EQUATIONS);
        
        seed = seed * 1103515245 + 12345;
        remainders[seed % NUM_EQUATIONS] = (remainders[seed % NUM_EQUATIONS] + 1) % moduli[seed % NUM_EQUATIONS];
    }
    
    long long product = 1;
    for (int i = 0; i < NUM_EQUATIONS; i++) {
        product *= moduli[i];
    }
    
    int verified = 1;
    for (int i = 0; i < NUM_EQUATIONS; i++) {
        if (results[0] % moduli[i] != remainders[i]) {
            verified = 0;
            break;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Chinese Remainder Theorem: %d equations, 100 tests, %.6f seconds\n",
           NUM_EQUATIONS, time_spent);
    printf("Solution: %lld (mod %lld)\n", results[0], product);
    printf("Verified: %s\n", verified ? "YES" : "NO");
    
    return 0;
}
