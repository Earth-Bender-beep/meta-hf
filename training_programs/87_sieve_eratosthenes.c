// Sieve of Eratosthenes for finding prime numbers
// Number theory, array marking, nested loops
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 2000000

void sieve_of_eratosthenes(int n, int *primes, int *count) {
    char *is_prime = (char*)malloc((n + 1) * sizeof(char));
    memset(is_prime, 1, n + 1);
    
    is_prime[0] = is_prime[1] = 0;
    
    for (int i = 2; i * i <= n; i++) {
        if (is_prime[i]) {
            for (int j = i * i; j <= n; j += i) {
                is_prime[j] = 0;
            }
        }
    }
    
    *count = 0;
    for (int i = 2; i <= n; i++) {
        if (is_prime[i]) {
            (*count)++;
        }
    }
    
    free(is_prime);
}

// Segmented sieve for larger ranges
void segmented_sieve(int n, int *count) {
    int limit = n;
    int segment_size = 32768;
    
    // Find primes up to sqrt(n)
    int sqrt_n = 1;
    while (sqrt_n * sqrt_n <= n) sqrt_n++;
    
    char *is_prime_small = (char*)malloc((sqrt_n + 1) * sizeof(char));
    memset(is_prime_small, 1, sqrt_n + 1);
    is_prime_small[0] = is_prime_small[1] = 0;
    
    for (int i = 2; i * i <= sqrt_n; i++) {
        if (is_prime_small[i]) {
            for (int j = i * i; j <= sqrt_n; j += i) {
                is_prime_small[j] = 0;
            }
        }
    }
    
    int *primes = (int*)malloc(sqrt_n * sizeof(int));
    int prime_count = 0;
    for (int i = 2; i <= sqrt_n; i++) {
        if (is_prime_small[i]) {
            primes[prime_count++] = i;
        }
    }
    
    *count = prime_count;
    
    // Process segments
    char *is_prime = (char*)malloc(segment_size * sizeof(char));
    
    for (int low = sqrt_n + 1; low <= limit; low += segment_size) {
        int high = low + segment_size - 1;
        if (high > limit) high = limit;
        
        memset(is_prime, 1, segment_size);
        
        for (int i = 0; i < prime_count; i++) {
            int p = primes[i];
            int start = ((low + p - 1) / p) * p;
            
            for (int j = start; j <= high; j += p) {
                is_prime[j - low] = 0;
            }
        }
        
        for (int i = 0; i < high - low + 1; i++) {
            if (is_prime[i]) (*count)++;
        }
    }
    
    free(is_prime);
    free(primes);
    free(is_prime_small);
}

int main() {
    int count1, count2;
    
    clock_t start = clock();
    sieve_of_eratosthenes(N, NULL, &count1);
    segmented_sieve(N, &count2);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Sieve of Eratosthenes: n=%d, %.6f seconds\n", N, time_spent);
    printf("Primes found: %d (standard), %d (segmented)\n", count1, count2);
    
    return 0;
}
