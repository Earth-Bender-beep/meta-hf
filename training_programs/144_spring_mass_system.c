// Spring-mass system with damping simulation
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_MASSES 100
#define NUM_STEPS 1000
#define DT 0.01
#define SPRING_K 10.0
#define DAMPING 0.5

typedef struct {
    double position;
    double velocity;
    double force;
    double mass;
} Mass;

void compute_forces(Mass *masses, int n, double k, double damping) {
    for (int i = 0; i < n; i++) {
        masses[i].force = 0.0;
    }
    
    // Springs between adjacent masses
    for (int i = 0; i < n - 1; i++) {
        double dx = masses[i + 1].position - masses[i].position;
        double dv = masses[i + 1].velocity - masses[i].velocity;
        
        double spring_force = k * dx;
        double damping_force = damping * dv;
        
        masses[i].force += spring_force + damping_force;
        masses[i + 1].force -= spring_force + damping_force;
    }
    
    // Gravity
    for (int i = 0; i < n; i++) {
        masses[i].force -= 9.8 * masses[i].mass;
    }
    
    // Fix first mass (anchor point)
    masses[0].force = 0.0;
}

void integrate_euler(Mass *masses, int n, double dt) {
    for (int i = 0; i < n; i++) {
        double acceleration = masses[i].force / masses[i].mass;
        masses[i].velocity += acceleration * dt;
        masses[i].position += masses[i].velocity * dt;
    }
}

double compute_total_energy(Mass *masses, int n, double k) {
    double kinetic = 0.0;
    double potential = 0.0;
    
    for (int i = 0; i < n; i++) {
        kinetic += 0.5 * masses[i].mass * masses[i].velocity * masses[i].velocity;
        potential += masses[i].mass * 9.8 * masses[i].position;
    }
    
    for (int i = 0; i < n - 1; i++) {
        double dx = masses[i + 1].position - masses[i].position;
        potential += 0.5 * k * dx * dx;
    }
    
    return kinetic + potential;
}

int main() {
    Mass *masses = (Mass*)malloc(NUM_MASSES * sizeof(Mass));
    
    for (int i = 0; i < NUM_MASSES; i++) {
        masses[i].position = (double)i;
        masses[i].velocity = 0.0;
        masses[i].force = 0.0;
        masses[i].mass = 1.0;
    }
    
    masses[NUM_MASSES / 2].velocity = 5.0;
    
    clock_t start = clock();
    
    double initial_energy = compute_total_energy(masses, NUM_MASSES, SPRING_K);
    
    for (int step = 0; step < NUM_STEPS; step++) {
        compute_forces(masses, NUM_MASSES, SPRING_K, DAMPING);
        integrate_euler(masses, NUM_MASSES, DT);
    }
    
    double final_energy = compute_total_energy(masses, NUM_MASSES, SPRING_K);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Spring-mass system: %d masses, %d steps, %.6f seconds\n",
           NUM_MASSES, NUM_STEPS, time_spent);
    printf("Energy: initial=%.2f, final=%.2f\n", initial_energy, final_energy);
    
    free(masses);
    
    return 0;
}
