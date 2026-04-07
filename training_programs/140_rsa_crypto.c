// RSA cryptography basics - modular exponentiation and simple encryption
// Educational implementation of RSA concepts
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_OPERATIONS 1000

// Modular exponentiation: (base^exp) % mod
unsigned long long mod_exp(unsigned long long base, unsigned long long exp, unsigned long long mod) {
    unsigned long long result = 1;
    base = base % mod;
    
    while (exp > 0) {
        if (exp & 1) {
            result = (result * base) % mod;
        }
        exp = exp >> 1;
        base = (base * base) % mod;
    }
    
    return result;
}

// Extended Euclidean Algorithm
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

// Compute modular multiplicative inverse
unsigned long long mod_inverse(unsigned long long a, unsigned long long m) {
    long long x, y;
    long long gcd = extended_gcd(a, m, &x, &y);
    
    if (gcd != 1) {
        return 0;  // Inverse doesn't exist
    }
    
    return (x % m + m) % m;
}

// Check if number is prime (Miller-Rabin test simplified)
int is_prime_simple(unsigned long long n) {
    if (n <= 1) return 0;
    if (n <= 3) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    
    for (unsigned long long i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) {
            return 0;
        }
    }
    
    return 1;
}

// Generate small primes for demonstration
unsigned long long find_prime(unsigned long long start) {
    for (unsigned long long n = start; n < start + 1000; n++) {
        if (is_prime_simple(n)) {
            return n;
        }
    }
    return 0;
}

typedef struct {
    unsigned long long n;  // Modulus
    unsigned long long e;  // Public exponent
    unsigned long long d;  // Private exponent
} RSAKeys;

void generate_rsa_keys(RSAKeys *keys, unsigned long long p, unsigned long long q) {
    keys->n = p * q;
    unsigned long long phi = (p - 1) * (q - 1);
    
    // Use common public exponent
    keys->e = 65537;
    
    // Check if e and phi are coprime
    long long x, y;
    if (extended_gcd(keys->e, phi, &x, &y) != 1) {
        keys->e = 17;  // Fallback
    }
    
    // Compute private exponent
    keys->d = mod_inverse(keys->e, phi);
}

unsigned long long rsa_encrypt(unsigned long long message, unsigned long long e, unsigned long long n) {
    return mod_exp(message, e, n);
}

unsigned long long rsa_decrypt(unsigned long long ciphertext, unsigned long long d, unsigned long long n) {
    return mod_exp(ciphertext, d, n);
}

int main() {
    // Use small primes for demonstration
    unsigned long long p = find_prime(1000);
    unsigned long long q = find_prime(2000);
    
    RSAKeys keys;
    generate_rsa_keys(&keys, p, q);
    
    printf("RSA Cryptography Demo\n");
    printf("p=%llu, q=%llu, n=%llu\n", p, q, keys.n);
    printf("Public key (e, n): (%llu, %llu)\n", keys.e, keys.n);
    printf("Private key (d, n): (%llu, %llu)\n", keys.d, keys.n);
    
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    int successful = 0;
    unsigned long long checksum = 0;
    
    for (int i = 0; i < NUM_OPERATIONS; i++) {
        seed = seed * 1103515245 + 12345;
        unsigned long long message = (seed % (keys.n - 100)) + 1;
        
        // Encrypt
        unsigned long long ciphertext = rsa_encrypt(message, keys.e, keys.n);
        
        // Decrypt
        unsigned long long decrypted = rsa_decrypt(ciphertext, keys.d, keys.n);
        
        if (decrypted == message) {
            successful++;
        }
        
        checksum += ciphertext;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("\nPerformance: %d operations, %.6f seconds\n", NUM_OPERATIONS, time_spent);
    printf("Successful encryptions/decryptions: %d/%d\n", successful, NUM_OPERATIONS);
    printf("Checksum: %llu\n", checksum);
    
    return 0;
}
