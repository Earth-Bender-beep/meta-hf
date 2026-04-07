// Sieve of Eratosthenes for prime number generation
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_N 10000000

void sieve_of_eratosthenes(int n, int *primes, int *count) {
    unsigned char *is_prime = (unsigned char*)malloc((n + 1) * sizeof(unsigned char));
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
            primes[(*count)++] = i;
        }
    }
    
    free(is_prime);
}

void segmented_sieve(int n, int *count) {
    int limit = sqrt(n) + 1;
    int *small_primes = (int*)malloc(limit * sizeof(int));
    int small_count;
    
    sieve_of_eratosthenes(limit, small_primes, &small_count);
    
    int segment_size = limit;
    unsigned char *is_prime = (unsigned char*)malloc(segment_size);
    
    *count = small_count;
    
    for (int low = limit; low <= n; low += segment_size) {
        memset(is_prime, 1, segment_size);
        
        int high = low + segment_size - 1;
        if (high > n) high = n;
        
        for (int i = 0; i < small_count; i++) {
            int prime = small_primes[i];
            int start = ((low + prime - 1) / prime) * prime;
            
            for (int j = start; j <= high; j += prime) {
                is_prime[j - low] = 0;
            }
        }
        
        for (int i = low; i <= high; i++) {
            if (is_prime[i - low]) {
                (*count)++;
            }
        }
    }
    
    free(small_primes);
    free(is_prime);
}

int main() {
    int n = 1000000;
    int *primes = (int*)malloc(n * sizeof(int));
    int count;
    
    clock_t start = clock();
    
    sieve_of_eratosthenes(n, primes, &count);
    
    int seg_count;
    segmented_sieve(n, &seg_count);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Prime sieve: n=%d, %.6f seconds\n", n, time_spent);
    printf("Primes found: %d (standard), %d (segmented)\n", count, seg_count);
    printf("Largest prime: %d\n", primes[count - 1]);
    
    free(primes);
    
    return 0;
}
