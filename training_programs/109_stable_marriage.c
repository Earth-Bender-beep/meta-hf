// Stable Marriage Problem - Gale-Shapley algorithm
// Matching algorithm with stability guarantees
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 500

int men_pref[N][N];      // Men's preference lists
int women_pref[N][N];    // Women's preference lists
int women_rank[N][N];    // Rank of each man in woman's preference
int men_partner[N];      // Current partner of each man (-1 if none)
int women_partner[N];    // Current partner of each woman (-1 if none)
int next_proposal[N];    // Next woman to propose to for each man

void init_preferences(int n) {
    unsigned int seed = 42;
    
    // Generate random preferences
    for (int i = 0; i < n; i++) {
        // Initialize men's preferences
        for (int j = 0; j < n; j++) {
            men_pref[i][j] = j;
        }
        
        // Shuffle using Fisher-Yates
        for (int j = n - 1; j > 0; j--) {
            seed = seed * 1103515245 + 12345;
            int k = seed % (j + 1);
            int temp = men_pref[i][j];
            men_pref[i][j] = men_pref[i][k];
            men_pref[i][k] = temp;
        }
        
        // Initialize women's preferences
        for (int j = 0; j < n; j++) {
            women_pref[i][j] = j;
        }
        
        // Shuffle
        for (int j = n - 1; j > 0; j--) {
            seed = seed * 1103515245 + 12345;
            int k = seed % (j + 1);
            int temp = women_pref[i][j];
            women_pref[i][j] = women_pref[i][k];
            women_pref[i][k] = temp;
        }
    }
    
    // Build rank matrix for women (for O(1) comparison)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            women_rank[i][women_pref[i][j]] = j;
        }
    }
}

void gale_shapley(int n) {
    // Initialize all as unmatched
    for (int i = 0; i < n; i++) {
        men_partner[i] = -1;
        women_partner[i] = -1;
        next_proposal[i] = 0;
    }
    
    int free_men = n;
    
    while (free_men > 0) {
        // Find a free man
        int man = -1;
        for (int i = 0; i < n; i++) {
            if (men_partner[i] == -1) {
                man = i;
                break;
            }
        }
        
        if (man == -1) break;
        
        // Man proposes to next woman on his list
        int woman = men_pref[man][next_proposal[man]];
        next_proposal[man]++;
        
        if (women_partner[woman] == -1) {
            // Woman is free, engage them
            men_partner[man] = woman;
            women_partner[woman] = man;
            free_men--;
        } else {
            // Woman is engaged, check if she prefers new man
            int current_partner = women_partner[woman];
            
            if (women_rank[woman][man] < women_rank[woman][current_partner]) {
                // Woman prefers new man
                men_partner[current_partner] = -1;
                men_partner[man] = woman;
                women_partner[woman] = man;
            }
        }
    }
}

int verify_stability(int n) {
    // Check if any pair would prefer each other over current partners
    for (int m = 0; m < n; m++) {
        for (int w = 0; w < n; w++) {
            if (men_partner[m] == w) continue;
            
            // Check if m prefers w over his current partner
            int m_prefers_w = 0;
            for (int i = 0; i < n; i++) {
                if (men_pref[m][i] == w) {
                    m_prefers_w = 1;
                    break;
                }
                if (men_pref[m][i] == men_partner[m]) break;
            }
            
            if (!m_prefers_w) continue;
            
            // Check if w prefers m over her current partner
            if (women_rank[w][m] < women_rank[w][women_partner[w]]) {
                return 0; // Unstable pair found
            }
        }
    }
    
    return 1; // Stable matching
}

int main() {
    int n = N;
    
    init_preferences(n);
    
    clock_t start = clock();
    gale_shapley(n);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    int stable = verify_stability(n);
    
    printf("Stable Marriage (Gale-Shapley): %d pairs, %.6f seconds\n",
           n, time_spent);
    printf("Matching is %s\n", stable ? "STABLE" : "UNSTABLE");
    
    return 0;
}
