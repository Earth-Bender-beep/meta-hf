// Particle Swarm Optimization
// Swarm intelligence, collective behavior
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_PARTICLES 50
#define DIMENSIONS 10
#define MAX_ITERATIONS 500
#define W 0.7  // Inertia weight
#define C1 1.5 // Cognitive parameter
#define C2 1.5 // Social parameter

typedef struct {
    double position[DIMENSIONS];
    double velocity[DIMENSIONS];
    double best_position[DIMENSIONS];
    double best_fitness;
} Particle;

// Rastrigin function (minimization)
double fitness_function(double *position) {
    double sum = 10.0 * DIMENSIONS;
    for (int i = 0; i < DIMENSIONS; i++) {
        sum += position[i] * position[i] - 10.0 * cos(2.0 * M_PI * position[i]);
    }
    return sum;
}

void init_particle(Particle *p, unsigned int *seed) {
    for (int i = 0; i < DIMENSIONS; i++) {
        *seed = *seed * 1103515245 + 12345;
        double rand_val = ((*seed) & 0x7FFFFFFF) / (double)0x7FFFFFFF;
        p->position[i] = -5.12 + rand_val * 10.24;  // Range [-5.12, 5.12]
        
        *seed = *seed * 1103515245 + 12345;
        rand_val = ((*seed) & 0x7FFFFFFF) / (double)0x7FFFFFFF;
        p->velocity[i] = -1.0 + rand_val * 2.0;
        
        p->best_position[i] = p->position[i];
    }
    p->best_fitness = fitness_function(p->position);
}

void update_particle(Particle *p, double *global_best, unsigned int *seed) {
    for (int i = 0; i < DIMENSIONS; i++) {
        *seed = *seed * 1103515245 + 12345;
        double r1 = ((*seed) & 0x7FFFFFFF) / (double)0x7FFFFFFF;
        
        *seed = *seed * 1103515245 + 12345;
        double r2 = ((*seed) & 0x7FFFFFFF) / (double)0x7FFFFFFF;
        
        // Update velocity
        p->velocity[i] = W * p->velocity[i] +
                        C1 * r1 * (p->best_position[i] - p->position[i]) +
                        C2 * r2 * (global_best[i] - p->position[i]);
        
        // Limit velocity
        if (p->velocity[i] > 2.0) p->velocity[i] = 2.0;
        if (p->velocity[i] < -2.0) p->velocity[i] = -2.0;
        
        // Update position
        p->position[i] += p->velocity[i];
        
        // Boundary constraints
        if (p->position[i] > 5.12) p->position[i] = 5.12;
        if (p->position[i] < -5.12) p->position[i] = -5.12;
    }
    
    double fitness = fitness_function(p->position);
    if (fitness < p->best_fitness) {
        p->best_fitness = fitness;
        for (int i = 0; i < DIMENSIONS; i++) {
            p->best_position[i] = p->position[i];
        }
    }
}

double particle_swarm_optimization(unsigned int *seed) {
    Particle *swarm = (Particle*)malloc(NUM_PARTICLES * sizeof(Particle));
    double global_best[DIMENSIONS];
    double global_best_fitness = 1e10;
    
    // Initialize swarm
    for (int i = 0; i < NUM_PARTICLES; i++) {
        init_particle(&swarm[i], seed);
        
        if (swarm[i].best_fitness < global_best_fitness) {
            global_best_fitness = swarm[i].best_fitness;
            for (int j = 0; j < DIMENSIONS; j++) {
                global_best[j] = swarm[i].best_position[j];
            }
        }
    }
    
    // Main PSO loop
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        for (int i = 0; i < NUM_PARTICLES; i++) {
            update_particle(&swarm[i], global_best, seed);
            
            if (swarm[i].best_fitness < global_best_fitness) {
                global_best_fitness = swarm[i].best_fitness;
                for (int j = 0; j < DIMENSIONS; j++) {
                    global_best[j] = swarm[i].best_position[j];
                }
            }
        }
    }
    
    free(swarm);
    return global_best_fitness;
}

int main() {
    unsigned int seed = 42;
    
    clock_t start = clock();
    double best_fitness = particle_swarm_optimization(&seed);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Particle Swarm Optimization: %d particles, %d iterations, %.6f seconds\n",
           NUM_PARTICLES, MAX_ITERATIONS, time_spent);
    printf("Best fitness found: %.6f (optimum is 0.0)\n", best_fitness);
    
    return 0;
}
