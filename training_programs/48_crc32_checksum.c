// CRC-32 checksum calculation
// Bitwise operations, table lookups, polynomial arithmetic
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define POLYNOMIAL 0xEDB88320
#define DATA_SIZE 100000

uint32_t crc32_table[256];

void generate_crc32_table() {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ POLYNOMIAL;
            } else {
                crc >>= 1;
            }
        }
        crc32_table[i] = crc;
    }
}

uint32_t crc32_calculate(const uint8_t *data, size_t length) {
    uint32_t crc = 0xFFFFFFFF;
    
    for (size_t i = 0; i < length; i++) {
        uint8_t index = (crc ^ data[i]) & 0xFF;
        crc = (crc >> 8) ^ crc32_table[index];
    }
    
    return ~crc;
}

// Bitwise version without table
uint32_t crc32_bitwise(const uint8_t *data, size_t length) {
    uint32_t crc = 0xFFFFFFFF;
    
    for (size_t i = 0; i < length; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ POLYNOMIAL;
            } else {
                crc >>= 1;
            }
        }
    }
    
    return ~crc;
}

void generate_test_data(uint8_t *data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = (uint8_t)(i * 17 + 23) % 256;
    }
}

int main() {
    uint8_t *data = (uint8_t*)malloc(DATA_SIZE);
    generate_test_data(data, DATA_SIZE);
    
    generate_crc32_table();
    
    clock_t start = clock();
    
    uint32_t crc1 = crc32_calculate(data, DATA_SIZE);
    uint32_t crc2 = crc32_bitwise(data, DATA_SIZE / 10);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("CRC32 checksum: %d bytes, %.6f seconds\n", DATA_SIZE, time_spent);
    printf("Table-based: 0x%08X, Bitwise: 0x%08X\n", crc1, crc2);
    
    free(data);
    return 0;
}
