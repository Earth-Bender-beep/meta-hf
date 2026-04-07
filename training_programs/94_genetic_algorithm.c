// Genetic algorithm for optimization
// Evolutionary computation, population-based search
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define POPULATION_SIZE 200
#define GENE_LENGTH 20
#define GENERATIONS 300
#define MUTATION_RATE 0.05
#define CROSSOVER_RATE 0.7

typedef struct {
    int genes[GENE_LENGTH];
    double fitness;
} Individual;

double evaluate_fitness(Individual *ind) {
    int sum = 0;
    for (int i = 0; i < GENE_LENGTH; i++) {
        sum += ind->genes[i];
    }
    // Target: maximize sum (simple example)
    return (double)sum;
}

void init_population(Individual *pop, unsigned int *seed) {
    for (int i = 0; i < POPULATION_SIZE; i++) {
        for (int j = 0; j < GENE_LENGTH; j++) {
            *seed = *seed * 1103515245 + 12345;
            pop[i].genes[j] = (*seed) % 2;
        }
        pop[i].fitness = evaluate_fitness(&pop[i]);
    }
}

Individual* tournament_selection(Individual *pop, unsigned int *seed) {
    int idx1 = (*seed = *seed * 1103515245 + 12345) % POPULATION_SIZE;
    int idx2 = (*seed = *seed * 1103515245 + 12345) % POPULATION_SIZE;
    
    return (pop[idx1].fitness > pop[idx2].fitness) ? &pop[idx1] : &pop[idx2];
}

void crossover(Individual *parent1, Individual *parent2, 
               Individual *child1, Individual *child2, unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    int crossover_point = (*seed) % GENE_LENGTH;
    
    for (int i = 0; i < GENE_LENGTH; i++) {
        if (i < crossover_point) {
            child1->genes[i] = parent1->genes[i];
            child2->genes[i] = parent2->genes[i];
        } else {
            child1->genes[i] = parent2->genes[i];
            child2->genes[i] = parent1->genes[i];
        }
    }
}

void mutate(Individual *ind, unsigned int *seed) {
    for (int i = 0; i < GENE_LENGTH; i++) {
        *seed = *seed * 1103515245 + 12345;
        double rand_val = ((*seed) & 0x7FFFFFFF) / (double)0x7FFFFFFF;
        
        if (rand_val < MUTATION_RATE) {
            ind->genes[i] = 1 - ind->genes[i];
        }
    }
}

void evolve_population(Individual *pop, Individual *new_pop, unsigned int *seed) {
    for (int i = 0; i < POPULATION_SIZE; i += 2) {
        Individual *parent1 = tournament_selection(pop, seed);
        Individual *parent2 = tournament_selection(pop, seed);
        
        *seed = *seed * 1103515245 + 12345;
        double rand_val = ((*seed) & 0x7FFFFFFF) / (double)0x7FFFFFFF;
        
        if (rand_val < CROSSOVER_RATE) {
            crossover(parent1, parent2, &new_pop[i], &new_pop[i+1], seed);
        } else {
            new_pop[i] = *parent1;
            new_pop[i+1] = *parent2;
        }
        
        mutate(&new_pop[i], seed);
        mutate(&new_pop[i+1], seed);
        
        new_pop[i].fitness = evaluate_fitness(&new_pop[i]);
        new_pop[i+1].fitness = evaluate_fitness(&new_pop[i+1]);
    }
}

double get_best_fitness(Individual *pop) {
    double best = pop[0].fitness;
    for (int i = 1; i < POPULATION_SIZE; i++) {
        if (pop[i].fitness > best) {
            best = pop[i].fitness;
        }
    }
    return best;
}

int main() {
    unsigned int seed = 42;
    Individual *population = (Individual*)malloc(POPULATION_SIZE * sizeof(Individual));
    Individual *new_population = (Individual*)malloc(POPULATION_SIZE * sizeof(Individual));
    
    init_population(population, &seed);
    
    clock_t start = clock();
    
    double initial_best = get_best_fitness(population);
    
    for (int gen = 0; gen < GENERATIONS; gen++) {
        evolve_population(population, new_population, &seed);
        
        Individual *temp = population;
        population = new_population;
        new_population = temp;
    }
    
    double final_best = get_best_fitness(population);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Genetic algorithm: %d generations, pop=%d, %.6f seconds\n",
           GENERATIONS, POPULATION_SIZE, time_spent);
    printf("Initial best: %.1f, Final best: %.1f\n", initial_best, final_best);
    
    free(population);
    free(new_population);
    
    return 0;
}
