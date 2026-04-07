// Lorenz attractor - chaotic dynamical system simulation
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_STEPS 50000
#define DT 0.01

typedef struct {
    double x, y, z;
} Point3D;

void lorenz_step(Point3D *p, double sigma, double rho, double beta, double dt) {
    double dx = sigma * (p->y - p->x);
    double dy = p->x * (rho - p->z) - p->y;
    double dz = p->x * p->y - beta * p->z;
    
    p->x += dx * dt;
    p->y += dy * dt;
    p->z += dz * dt;
}

double compute_distance(Point3D *a, Point3D *b) {
    double dx = a->x - b->x;
    double dy = a->y - b->y;
    double dz = a->z - b->z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

int main() {
    Point3D *trajectory = (Point3D*)malloc(NUM_STEPS * sizeof(Point3D));
    
    trajectory[0].x = 1.0;
    trajectory[0].y = 1.0;
    trajectory[0].z = 1.0;
    
    double sigma = 10.0;
    double rho = 28.0;
    double beta = 8.0 / 3.0;
    
    clock_t start = clock();
    
    for (int i = 1; i < NUM_STEPS; i++) {
        trajectory[i] = trajectory[i - 1];
        lorenz_step(&trajectory[i], sigma, rho, beta, DT);
    }
    
    double max_dist = 0.0;
    for (int i = 1000; i < NUM_STEPS; i += 100) {
        double dist = compute_distance(&trajectory[i], &trajectory[0]);
        if (dist > max_dist) {
            max_dist = dist;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Lorenz attractor: %d steps, dt=%.4f, %.6f seconds\n",
           NUM_STEPS, DT, time_spent);
    printf("Max distance from origin: %.2f\n", max_dist);
    
    free(trajectory);
    
    return 0;
}
