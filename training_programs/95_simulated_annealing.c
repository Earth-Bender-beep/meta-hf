// Simulated annealing for optimization
// Probabilistic optimization, temperature schedule
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PROBLEM_SIZE 50
#define MAX_ITERATIONS 50000
#define INITIAL_TEMP 100.0
#define COOLING_RATE 0.9999

typedef struct {
    int values[PROBLEM_SIZE];
    double cost;
} Solution;

double calculate_cost(Solution *sol) {
    // Example: traveling salesman-like cost
    int total = 0;
    for (int i = 0; i < PROBLEM_SIZE - 1; i++) {
        int diff = abs(sol->values[i] - sol->values[i+1]);
        total += diff * diff;
    }
    return (double)total;
}

void init_solution(Solution *sol, unsigned int *seed) {
    for (int i = 0; i < PROBLEM_SIZE; i++) {
        sol->values[i] = i;
    }
    
    // Shuffle
    for (int i = PROBLEM_SIZE - 1; i > 0; i--) {
        *seed = *seed * 1103515245 + 12345;
        int j = (*seed) % (i + 1);
        int temp = sol->values[i];
        sol->values[i] = sol->values[j];
        sol->values[j] = temp;
    }
    
    sol->cost = calculate_cost(sol);
}

void generate_neighbor(Solution *current, Solution *neighbor, unsigned int *seed) {
    // Copy current solution
    for (int i = 0; i < PROBLEM_SIZE; i++) {
        neighbor->values[i] = current->values[i];
    }
    
    // Swap two random elements
    *seed = *seed * 1103515245 + 12345;
    int i = (*seed) % PROBLEM_SIZE;
    *seed = *seed * 1103515245 + 12345;
    int j = (*seed) % PROBLEM_SIZE;
    
    int temp = neighbor->values[i];
    neighbor->values[i] = neighbor->values[j];
    neighbor->values[j] = temp;
    
    neighbor->cost = calculate_cost(neighbor);
}

double simulated_annealing(Solution *best, unsigned int *seed) {
    Solution current, neighbor;
    
    init_solution(&current, seed);
    *best = current;
    
    double temperature = INITIAL_TEMP;
    
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        generate_neighbor(&current, &neighbor, seed);
        
        double delta = neighbor.cost - current.cost;
        
        if (delta < 0) {
            current = neighbor;
            if (current.cost < best->cost) {
                *best = current;
            }
        } else {
            *seed = *seed * 1103515245 + 12345;
            double rand_val = ((*seed) & 0x7FFFFFFF) / (double)0x7FFFFFFF;
            double acceptance_prob = exp(-delta / temperature);
            
            if (rand_val < acceptance_prob) {
                current = neighbor;
            }
        }
        
        temperature *= COOLING_RATE;
    }
    
    return best->cost;
}

int main() {
    unsigned int seed = 12345;
    Solution best;
    
    clock_t start = clock();
    double final_cost = simulated_annealing(&best, &seed);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Simulated annealing: %d iterations, %.6f seconds\n",
           MAX_ITERATIONS, time_spent);
    printf("Best cost found: %.2f\n", final_cost);
    
    return 0;
}
