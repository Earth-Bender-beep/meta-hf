// Apriori algorithm for frequent itemset mining
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_ITEMS 20
#define NUM_TRANSACTIONS 1000
#define MIN_SUPPORT 50

typedef struct {
    int items[MAX_ITEMS];
    int count;
} Itemset;

typedef struct {
    int transaction[MAX_ITEMS];
    int size;
} Transaction;

int contains_itemset(Transaction *trans, Itemset *itemset) {
    for (int i = 0; i < itemset->count; i++) {
        int found = 0;
        for (int j = 0; j < trans->size; j++) {
            if (trans->transaction[j] == itemset->items[i]) {
                found = 1;
                break;
            }
        }
        if (!found) return 0;
    }
    return 1;
}

int count_support(Transaction *transactions, int num_trans, Itemset *itemset) {
    int support = 0;
    for (int i = 0; i < num_trans; i++) {
        if (contains_itemset(&transactions[i], itemset)) {
            support++;
        }
    }
    return support;
}

void generate_1_itemsets(Transaction *transactions, int num_trans, 
                        Itemset *frequent, int *freq_count, int min_support) {
    int item_support[MAX_ITEMS] = {0};
    
    for (int i = 0; i < num_trans; i++) {
        for (int j = 0; j < transactions[i].size; j++) {
            item_support[transactions[i].transaction[j]]++;
        }
    }
    
    *freq_count = 0;
    for (int i = 0; i < MAX_ITEMS; i++) {
        if (item_support[i] >= min_support) {
            frequent[*freq_count].items[0] = i;
            frequent[*freq_count].count = 1;
            (*freq_count)++;
        }
    }
}

void apriori(Transaction *transactions, int num_trans, int min_support) {
    Itemset *frequent = (Itemset*)malloc(1000 * sizeof(Itemset));
    int freq_count;
    
    generate_1_itemsets(transactions, num_trans, frequent, &freq_count, min_support);
    
    Itemset *candidates = (Itemset*)malloc(1000 * sizeof(Itemset));
    int candidate_count = 0;
    
    for (int i = 0; i < freq_count; i++) {
        for (int j = i + 1; j < freq_count; j++) {
            candidates[candidate_count].items[0] = frequent[i].items[0];
            candidates[candidate_count].items[1] = frequent[j].items[0];
            candidates[candidate_count].count = 2;
            
            int support = count_support(transactions, num_trans, &candidates[candidate_count]);
            if (support >= min_support) {
                candidate_count++;
            }
        }
    }
    
    free(frequent);
    free(candidates);
}

int main() {
    int num_trans = NUM_TRANSACTIONS;
    Transaction *transactions = (Transaction*)malloc(num_trans * sizeof(Transaction));
    
    unsigned int seed = 42;
    for (int i = 0; i < num_trans; i++) {
        seed = seed * 1103515245 + 12345;
        transactions[i].size = (seed % 10) + 3;
        
        for (int j = 0; j < transactions[i].size; j++) {
            seed = seed * 1103515245 + 12345;
            transactions[i].transaction[j] = seed % MAX_ITEMS;
        }
    }
    
    clock_t start = clock();
    
    apriori(transactions, num_trans, MIN_SUPPORT);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Apriori mining: %d transactions, min_support=%d, %.6f seconds\n",
           num_trans, MIN_SUPPORT, time_spent);
    
    free(transactions);
    
    return 0;
}
