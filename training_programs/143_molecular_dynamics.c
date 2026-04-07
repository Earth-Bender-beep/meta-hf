// Molecular dynamics simulation with Lennard-Jones potential
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_PARTICLES 500
#define NUM_STEPS 100
#define DT 0.001
#define BOX_SIZE 10.0

typedef struct {
    double x, y, z;
} Vector3D;

typedef struct {
    Vector3D position;
    Vector3D velocity;
    Vector3D force;
} Particle;

double lennard_jones_potential(double r) {
    double r6 = pow(r, 6);
    double r12 = r6 * r6;
    return 4.0 * (1.0 / r12 - 1.0 / r6);
}

void compute_forces(Particle *particles, int n) {
    for (int i = 0; i < n; i++) {
        particles[i].force.x = 0.0;
        particles[i].force.y = 0.0;
        particles[i].force.z = 0.0;
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dx = particles[i].position.x - particles[j].position.x;
            double dy = particles[i].position.y - particles[j].position.y;
            double dz = particles[i].position.z - particles[j].position.z;
            
            // Apply periodic boundary conditions
            if (dx > BOX_SIZE / 2) dx -= BOX_SIZE;
            if (dx < -BOX_SIZE / 2) dx += BOX_SIZE;
            if (dy > BOX_SIZE / 2) dy -= BOX_SIZE;
            if (dy < -BOX_SIZE / 2) dy += BOX_SIZE;
            if (dz > BOX_SIZE / 2) dz -= BOX_SIZE;
            if (dz < -BOX_SIZE / 2) dz += BOX_SIZE;
            
            double r2 = dx * dx + dy * dy + dz * dz;
            double r = sqrt(r2);
            
            if (r < 2.5 && r > 0.1) {
                double r6 = r2 * r2 * r2;
                double r8 = r6 * r2;
                double r14 = r6 * r6 * r2;
                
                double f_magnitude = 24.0 * (2.0 / r14 - 1.0 / r8);
                
                double fx = f_magnitude * dx / r;
                double fy = f_magnitude * dy / r;
                double fz = f_magnitude * dz / r;
                
                particles[i].force.x += fx;
                particles[i].force.y += fy;
                particles[i].force.z += fz;
                
                particles[j].force.x -= fx;
                particles[j].force.y -= fy;
                particles[j].force.z -= fz;
            }
        }
    }
}

void integrate_verlet(Particle *particles, int n, double dt) {
    for (int i = 0; i < n; i++) {
        particles[i].velocity.x += 0.5 * particles[i].force.x * dt;
        particles[i].velocity.y += 0.5 * particles[i].force.y * dt;
        particles[i].velocity.z += 0.5 * particles[i].force.z * dt;
        
        particles[i].position.x += particles[i].velocity.x * dt;
        particles[i].position.y += particles[i].velocity.y * dt;
        particles[i].position.z += particles[i].velocity.z * dt;
        
        // Periodic boundaries
        if (particles[i].position.x < 0) particles[i].position.x += BOX_SIZE;
        if (particles[i].position.x >= BOX_SIZE) particles[i].position.x -= BOX_SIZE;
        if (particles[i].position.y < 0) particles[i].position.y += BOX_SIZE;
        if (particles[i].position.y >= BOX_SIZE) particles[i].position.y -= BOX_SIZE;
        if (particles[i].position.z < 0) particles[i].position.z += BOX_SIZE;
        if (particles[i].position.z >= BOX_SIZE) particles[i].position.z -= BOX_SIZE;
    }
    
    compute_forces(particles, n);
    
    for (int i = 0; i < n; i++) {
        particles[i].velocity.x += 0.5 * particles[i].force.x * dt;
        particles[i].velocity.y += 0.5 * particles[i].force.y * dt;
        particles[i].velocity.z += 0.5 * particles[i].force.z * dt;
    }
}

int main() {
    Particle *particles = (Particle*)malloc(NUM_PARTICLES * sizeof(Particle));
    
    unsigned int seed = 42;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        seed = seed * 1103515245 + 12345;
        particles[i].position.x = ((seed & 0xFFFF) / (double)0xFFFF) * BOX_SIZE;
        seed = seed * 1103515245 + 12345;
        particles[i].position.y = ((seed & 0xFFFF) / (double)0xFFFF) * BOX_SIZE;
        seed = seed * 1103515245 + 12345;
        particles[i].position.z = ((seed & 0xFFFF) / (double)0xFFFF) * BOX_SIZE;
        
        particles[i].velocity.x = 0.0;
        particles[i].velocity.y = 0.0;
        particles[i].velocity.z = 0.0;
    }
    
    clock_t start = clock();
    
    compute_forces(particles, NUM_PARTICLES);
    
    for (int step = 0; step < NUM_STEPS; step++) {
        integrate_verlet(particles, NUM_PARTICLES, DT);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Molecular dynamics: %d particles, %d steps, %.6f seconds\n",
           NUM_PARTICLES, NUM_STEPS, time_spent);
    
    free(particles);
    
    return 0;
}
