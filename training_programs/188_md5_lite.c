// MD5 hash implementation (educational/lite version)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))

const uint32_t k[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};

void md5_hash(const uint8_t *msg, size_t len, uint32_t *digest) {
    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xefcdab89;
    uint32_t h2 = 0x98badcfe;
    uint32_t h3 = 0x10325476;
    
    size_t new_len = ((((len + 8) / 64) + 1) * 64) - 8;
    uint8_t *padded = (uint8_t*)calloc(new_len + 64, 1);
    memcpy(padded, msg, len);
    padded[len] = 0x80;
    
    uint64_t bits_len = len * 8;
    memcpy(padded + new_len, &bits_len, 8);
    
    for (size_t offset = 0; offset < new_len; offset += 64) {
        uint32_t *w = (uint32_t*)(padded + offset);
        
        uint32_t a = h0, b = h1, c = h2, d = h3;
        
        for (uint32_t i = 0; i < 64; i++) {
            uint32_t f, g;
            
            if (i < 16) {
                f = (b & c) | ((~b) & d);
                g = i;
            } else if (i < 32) {
                f = (d & b) | ((~d) & c);
                g = (5 * i + 1) % 16;
            } else if (i < 48) {
                f = b ^ c ^ d;
                g = (3 * i + 5) % 16;
            } else {
                f = c ^ (b | (~d));
                g = (7 * i) % 16;
            }
            
            uint32_t temp = d;
            d = c;
            c = b;
            b = b + LEFTROTATE((a + f + k[i] + w[g]), (uint32_t[]){7,12,17,22,5,9,14,20,4,11,16,23,6,10,15,21}[i % 16]);
            a = temp;
        }
        
        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
    }
    
    free(padded);
    
    digest[0] = h0;
    digest[1] = h1;
    digest[2] = h2;
    digest[3] = h3;
}

int main() {
    const char *messages[] = {
        "hello",
        "world",
        "test message",
        "MD5 hash function",
        "cryptographic hash"
    };
    
    clock_t start = clock();
    
    uint32_t digests[5][4];
    for (int test = 0; test < 10000; test++) {
        for (int i = 0; i < 5; i++) {
            md5_hash((uint8_t*)messages[i], strlen(messages[i]), digests[i]);
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("MD5 hash: 5 messages x 10000 iterations, %.6f seconds\n", time_spent);
    printf("Sample hash: %08x%08x%08x%08x\n", 
           digests[0][0], digests[0][1], digests[0][2], digests[0][3]);
    
    return 0;
}
