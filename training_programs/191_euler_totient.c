// Euler's totient function computation
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_N 100000

int euler_totient(int n) {
    int result = n;
    
    for (int p = 2; p * p <= n; p++) {
        if (n % p == 0) {
            while (n % p == 0) {
                n /= p;
            }
            result -= result / p;
        }
    }
    
    if (n > 1) {
        result -= result / n;
    }
    
    return result;
}

void compute_totient_sieve(int n, int *phi) {
    for (int i = 1; i <= n; i++) {
        phi[i] = i;
    }
    
    for (int i = 2; i <= n; i++) {
        if (phi[i] == i) {
            for (int j = i; j <= n; j += i) {
                phi[j] -= phi[j] / i;
            }
        }
    }
}

int main() {
    int n = 10000;
    int *phi = (int*)malloc((n + 1) * sizeof(int));
    
    clock_t start = clock();
    
    long long sum_single = 0;
    for (int i = 1; i <= n; i++) {
        sum_single += euler_totient(i);
    }
    
    compute_totient_sieve(n, phi);
    
    long long sum_sieve = 0;
    for (int i = 1; i <= n; i++) {
        sum_sieve += phi[i];
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Euler totient: n=%d, %.6f seconds\n", n, time_spent);
    printf("Sum single: %lld, Sum sieve: %lld\n", sum_single, sum_sieve);
    
    free(phi);
    return 0;
}
