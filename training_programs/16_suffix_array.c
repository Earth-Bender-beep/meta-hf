#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct Suffix {
    int index;
    char* suff;
} Suffix;

int cmp(const void* a, const void* b) {
    return strcmp(((Suffix*)a)->suff, ((Suffix*)b)->suff);
}

int* build_suffix_array(char* txt, int n) {
    Suffix* suffixes = (Suffix*)malloc(n * sizeof(Suffix));
    
    for (int i = 0; i < n; i++) {
        suffixes[i].index = i;
        suffixes[i].suff = txt + i;
    }
    
    qsort(suffixes, n, sizeof(Suffix), cmp);
    
    int* suffix_arr = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
        suffix_arr[i] = suffixes[i].index;
    
    free(suffixes);
    return suffix_arr;
}

int main() {
    int n = 10000;
    char* txt = (char*)malloc(n + 1);
    
    srand(42);
    for (int i = 0; i < n; i++) {
        txt[i] = 'a' + (rand() % 4);
    }
    txt[n] = '\0';
    
    clock_t start = clock();
    int* suffix_arr = build_suffix_array(txt, n);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Suffix array: n=%d in %.6f seconds\n", n, time_spent);
    
    free(txt);
    free(suffix_arr);
    return 0;
}
