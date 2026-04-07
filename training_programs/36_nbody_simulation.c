// N-body gravitational simulation
// Floating-point heavy, nested loops, low branching
#include <stdio.h>
<

#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_BODIES 256
#define TIME_STEPS 50
#define DT 0.01
#define G 6.674e-11
#define SOFTENING 1e-9

typedef struct {
    double x, y, z;
    double vx, vy, vz;
    double mass;
} Body;

void compute_forces(Body *bodies, int n, double *fx, double *fy, double *fz) {
    for (int i = 0; i < n; i++) {
        fx[i] = fy[i] = fz[i] = 0.0;
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double dz = bodies[j].z - bodies[i].z;
            
            double dist_sq = dx*dx + dy*dy + dz*dz + SOFTENING;
            double dist = sqrt(dist_sq);
            double force = G * bodies[i].mass * bodies[j].mass / dist_sq;
            
            double fx_comp = force * dx / dist;
            double fy_comp = force * dy / dist;
            double fz_comp = force * dz / dist;
            
            fx[i] += fx_comp;
            fy[i] += fy_comp;
            fz[i] += fz_comp;
            
            fx[j] -= fx_comp;
            fy[j] -= fy_comp;
            fz[j] -= fz_comp;
        }
    }
}

void update_positions(Body *bodies, int n, double *fx, double *fy, double *fz, double dt) {
    for (int i = 0; i < n; i++) {
        double ax = fx[i] / bodies[i].mass;
        double ay = fy[i] / bodies[i].mass;
        double az = fz[i] / bodies[i].mass;
        
        bodies[i].vx += ax * dt;
        bodies[i].vy += ay * dt;
        bodies[i].vz += az * dt;
        
        bodies[i].x += bodies[i].vx * dt;
        bodies[i].y += bodies[i].vy * dt;
        bodies[i].z += bodies[i].vz * dt;
    }
}

void init_bodies(Body *bodies, int n) {
    for (int i = 0; i < n; i++) {
        bodies[i].x = (double)(i % 16) * 10.0;
        bodies[i].y = (double)(i / 16) * 10.0;
        bodies[i].z = (double)(i % 7) * 5.0;
        bodies[i].vx = ((i * 13) % 100 - 50) / 100.0;
        bodies[i].vy = ((i * 17) % 100 - 50) / 100.0;
        bodies[i].vz = ((i * 19) % 100 - 50) / 100.0;
        bodies[i].mass = 1.0e20 + (i % 10) * 1.0e19;
    }
}

int main() {
    Body *bodies = (Body*)malloc(N_BODIES * sizeof(Body));
    double *fx = (double*)malloc(N_BODIES * sizeof(double));
    double *fy = (double*)malloc(N_BODIES * sizeof(double));
    double *fz = (double*)malloc(N_BODIES * sizeof(double));
    
    init_bodies(bodies, N_BODIES);
    
    clock_t start = clock();
    
    for (int step = 0; step < TIME_STEPS; step++) {
        compute_forces(bodies, N_BODIES, fx, fy, fz);
        update_positions(bodies, N_BODIES, fx, fy, fz, DT);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("N-body simulation: %d bodies, %d steps, %.6f seconds\n",
           N_BODIES, TIME_STEPS, time_spent);
    printf("Final position[0]: (%.2f, %.2f, %.2f)\n", 
           bodies[0].x, bodies[0].y, bodies[0].z);
    
    free(bodies);
    free(fx);
    free(fy);
    free(fz);
    return 0;
}
