// Modular exponentiation for cryptography
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_TESTS 50000

long long mod_exp_simple(long long base, long long exp, long long mod) {
    long long result = 1;
    base = base % mod;
    
    while (exp > 0) {
        if (exp % 2 == 1) {
            result = (result * base) % mod;
        }
        exp = exp / 2;
        base = (base * base) % mod;
    }
    
    return result;
}

long long mod_exp_binary(long long base, long long exp, long long mod) {
    long long result = 1;
    base = base % mod;
    
    while (exp > 0) {
        if (exp & 1) {
            result = (result * base) % mod;
        }
        base = (base * base) % mod;
        exp >>= 1;
    }
    
    return result;
}

long long mod_mult(long long a, long long b, long long mod) {
    long long result = 0;
    a = a % mod;
    
    while (b > 0) {
        if (b & 1) {
            result = (result + a) % mod;
        }
        a = (a * 2) % mod;
        b >>= 1;
    }
    
    return result;
}

int main() {
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    long long sum = 0;
    for (int i = 0; i < NUM_TESTS; i++) {
        seed = seed * 1103515245 + 12345;
        long long base = (seed % 1000) + 2;
        seed = seed * 1103515245 + 12345;
        long long exp = (seed % 1000) + 1;
        seed = seed * 1103515245 + 12345;
        long long mod = (seed % 10000) + 1000;
        
        sum += mod_exp_simple(base, exp, mod);
    }
    
    long long sum2 = 0;
    seed = 42;
    for (int i = 0; i < NUM_TESTS; i++) {
        seed = seed * 1103515245 + 12345;
        long long base = (seed % 1000) + 2;
        seed = seed * 1103515245 + 12345;
        long long exp = (seed % 1000) + 1;
        seed = seed * 1103515245 + 12345;
        long long mod = (seed % 10000) + 1000;
        
        sum2 += mod_exp_binary(base, exp, mod);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Modular exponentiation: %d tests, %.6f seconds\n", NUM_TESTS * 2, time_spent);
    printf("Sum simple: %lld, Sum binary: %lld\n", sum, sum2);
    
    return 0;
}
