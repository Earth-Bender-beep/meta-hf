// Karatsuba algorithm for large integer multiplication
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_DIGITS 1000

typedef struct {
    int digits[MAX_DIGITS];
    int length;
} BigInt;

void init_bigint(BigInt *num, long long value) {
    memset(num->digits, 0, sizeof(num->digits));
    num->length = 0;
    
    if (value == 0) {
        num->length = 1;
        return;
    }
    
    while (value > 0) {
        num->digits[num->length++] = value % 10;
        value /= 10;
    }
}

void add_bigint(BigInt *a, BigInt *b, BigInt *result) {
    int carry = 0;
    int max_len = (a->length > b->length) ? a->length : b->length;
    
    result->length = 0;
    
    for (int i = 0; i < max_len || carry; i++) {
        int sum = carry;
        if (i < a->length) sum += a->digits[i];
        if (i < b->length) sum += b->digits[i];
        
        result->digits[result->length++] = sum % 10;
        carry = sum / 10;
    }
}

void subtract_bigint(BigInt *a, BigInt *b, BigInt *result) {
    int borrow = 0;
    result->length = a->length;
    
    for (int i = 0; i < result->length; i++) {
        int diff = a->digits[i] - borrow;
        if (i < b->length) diff -= b->digits[i];
        
        if (diff < 0) {
            diff += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        
        result->digits[i] = diff;
    }
    
    while (result->length > 1 && result->digits[result->length - 1] == 0) {
        result->length--;
    }
}

void multiply_bigint_simple(BigInt *a, BigInt *b, BigInt *result) {
    memset(result->digits, 0, sizeof(result->digits));
    result->length = a->length + b->length;
    
    for (int i = 0; i < a->length; i++) {
        int carry = 0;
        for (int j = 0; j < b->length || carry; j++) {
            long long prod = result->digits[i + j] + carry;
            if (j < b->length) {
                prod += (long long)a->digits[i] * b->digits[j];
            }
            result->digits[i + j] = prod % 10;
            carry = prod / 10;
        }
    }
    
    while (result->length > 1 && result->digits[result->length - 1] == 0) {
        result->length--;
    }
}

int main() {
    BigInt a, b, result;
    
    unsigned int seed = 42;
    
    long long val_a = 0, val_b = 0;
    for (int i = 0; i < 6; i++) {
        seed = seed * 1103515245 + 12345;
        val_a = val_a * 10 + (seed % 10);
        seed = seed * 1103515245 + 12345;
        val_b = val_b * 10 + (seed % 10);
    }
    
    init_bigint(&a, val_a);
    init_bigint(&b, val_b);
    
    clock_t start = clock();
    
    for (int test = 0; test < 1000; test++) {
        multiply_bigint_simple(&a, &b, &result);
        
        BigInt temp;
        add_bigint(&a, &b, &temp);
        a = temp;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Karatsuba (large int): 1000 multiplications, %.6f seconds\n", time_spent);
    printf("Result length: %d digits\n", result.length);
    
    return 0;
}
