// Markov chain simulation and steady-state analysis
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_STATES 10
#define NUM_STEPS 100000

typedef struct {
    double transition[NUM_STATES][NUM_STATES];
    int current_state;
} MarkovChain;

void init_markov_chain(MarkovChain *mc, unsigned int *seed) {
    for (int i = 0; i < NUM_STATES; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < NUM_STATES; j++) {
            *seed = *seed * 1103515245 + 12345;
            mc->transition[i][j] = (*seed & 0xFFFF) / (double)0xFFFF;
            row_sum += mc->transition[i][j];
        }
        
        for (int j = 0; j < NUM_STATES; j++) {
            mc->transition[i][j] /= row_sum;
        }
    }
    
    mc->current_state = 0;
}

int next_state(MarkovChain *mc, unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    double r = (*seed & 0x7FFFFFFF) / (double)0x7FFFFFFF;
    
    double cumulative = 0.0;
    for (int j = 0; j < NUM_STATES; j++) {
        cumulative += mc->transition[mc->current_state][j];
        if (r <= cumulative) {
            mc->current_state = j;
            return j;
        }
    }
    
    return NUM_STATES - 1;
}

void simulate_markov_chain(MarkovChain *mc, int steps, int *state_counts, unsigned int *seed) {
    for (int i = 0; i < NUM_STATES; i++) {
        state_counts[i] = 0;
    }
    
    for (int step = 0; step < steps; step++) {
        int state = next_state(mc, seed);
        state_counts[state]++;
    }
}

int main() {
    MarkovChain mc;
    unsigned int seed = 42;
    
    init_markov_chain(&mc, &seed);
    
    int *state_counts = (int*)calloc(NUM_STATES, sizeof(int));
    
    clock_t start = clock();
    
    simulate_markov_chain(&mc, NUM_STEPS, state_counts, &seed);
    
    double *steady_state = (double*)malloc(NUM_STATES * sizeof(double));
    for (int i = 0; i < NUM_STATES; i++) {
        steady_state[i] = state_counts[i] / (double)NUM_STEPS;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Markov chain: %d states, %d steps, %.6f seconds\n",
           NUM_STATES, NUM_STEPS, time_spent);
    printf("Steady state probabilities: ");
    for (int i = 0; i < 5; i++) {
        printf("%.4f ", steady_state[i]);
    }
    printf("...\n");
    
    free(state_counts);
    free(steady_state);
    
    return 0;
}
