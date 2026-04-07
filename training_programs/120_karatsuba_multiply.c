// Karatsuba algorithm for fast integer multiplication
// Divide and conquer O(n^1.585) multiplication instead of O(n^2)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_DIGITS 2048
#define NUM_TESTS 5000

typedef struct {
    int digits[MAX_DIGITS];
    int len;
} BigInt;

void init_bigint(BigInt *num, long long val) {
    num->len = 0;
    if (val == 0) {
        num->digits[num->len++] = 0;
        return;
    }
    
    while (val > 0) {
        num->digits[num->len++] = val % 10;
        val /= 10;
    }
}

void add_bigint(BigInt *a, BigInt *b, BigInt *result) {
    int carry = 0;
    result->len = 0;
    int max_len = (a->len > b->len) ? a->len : b->len;
    
    for (int i = 0; i < max_len || carry; i++) {
        int digit_sum = carry;
        if (i < a->len) digit_sum += a->digits[i];
        if (i < b->len) digit_sum += b->digits[i];
        
        result->digits[result->len++] = digit_sum % 10;
        carry = digit_sum / 10;
    }
}

void subtract_bigint(BigInt *a, BigInt *b, BigInt *result) {
    int borrow = 0;
    result->len = 0;
    
    for (int i = 0; i < a->len; i++) {
        int diff = a->digits[i] - borrow;
        if (i < b->len) diff -= b->digits[i];
        
        if (diff < 0) {
            diff += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        
        result->digits[result->len++] = diff;
    }
    
    // Remove leading zeros
    while (result->len > 1 && result->digits[result->len - 1] == 0) {
        result->len--;
    }
}

void shift_left(BigInt *num, int places) {
    if (num->len == 1 && num->digits[0] == 0) return;
    
    for (int i = num->len - 1; i >= 0; i--) {
        num->digits[i + places] = num->digits[i];
    }
    for (int i = 0; i < places; i++) {
        num->digits[i] = 0;
    }
    num->len += places;
}

void karatsuba_multiply(BigInt *x, BigInt *y, BigInt *result) {
    // Base case: small numbers
    if (x->len <= 2 || y->len <= 2) {
        long long x_val = 0, y_val = 0;
        for (int i = x->len - 1; i >= 0; i--) {
            x_val = x_val * 10 + x->digits[i];
        }
        for (int i = y->len - 1; i >= 0; i--) {
            y_val = y_val * 10 + y->digits[i];
        }
        init_bigint(result, x_val * y_val);
        return;
    }
    
    int m = (x->len > y->len ? x->len : y->len) / 2;
    
    BigInt x0, x1, y0, y1;
    x0.len = x1.len = y0.len = y1.len = 0;
    
    // Split x and y
    for (int i = 0; i < m && i < x->len; i++) {
        x0.digits[x0.len++] = x->digits[i];
    }
    for (int i = m; i < x->len; i++) {
        x1.digits[x1.len++] = x->digits[i];
    }
    if (x1.len == 0) x1.digits[x1.len++] = 0;
    
    for (int i = 0; i < m && i < y->len; i++) {
        y0.digits[y0.len++] = y->digits[i];
    }
    for (int i = m; i < y->len; i++) {
        y1.digits[y1.len++] = y->digits[i];
    }
    if (y1.len == 0) y1.digits[y1.len++] = 0;
    
    // Three multiplications
    BigInt z0, z1, z2;
    karatsuba_multiply(&x0, &y0, &z0);
    karatsuba_multiply(&x1, &y1, &z2);
    
    BigInt x_sum, y_sum, z1_temp, z1_sub;
    add_bigint(&x0, &x1, &x_sum);
    add_bigint(&y0, &y1, &y_sum);
    karatsuba_multiply(&x_sum, &y_sum, &z1_temp);
    subtract_bigint(&z1_temp, &z0, &z1_sub);
    subtract_bigint(&z1_sub, &z2, &z1);
    
    // Combine results
    shift_left(&z2, 2 * m);
    shift_left(&z1, m);
    
    BigInt temp;
    add_bigint(&z2, &z1, &temp);
    add_bigint(&temp, &z0, result);
}

int main() {
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    long long checksum = 0;
    
    for (int test = 0; test < NUM_TESTS; test++) {
        seed = seed * 1103515245 + 12345;
        long long a = (seed % 999999) + 1;
        seed = seed * 1103515245 + 12345;
        long long b = (seed % 999999) + 1;
        
        BigInt x, y, result;
        init_bigint(&x, a);
        init_bigint(&y, b);
        karatsuba_multiply(&x, &y, &result);
        
        // Sum up result digits for checksum
        for (int i = 0; i < result.len; i++) {
            checksum += result.digits[i];
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Karatsuba multiplication: %d tests, %.6f seconds\n",
           NUM_TESTS, time_spent);
    printf("Digit checksum: %lld\n", checksum);
    
    return 0;
}
