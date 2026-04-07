// Collision detection - AABB and sphere collision algorithms
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_OBJECTS 500

typedef struct {
    double x, y, z;
} Vector3D;

typedef struct {
    Vector3D min;
    Vector3D max;
} AABB;

typedef struct {
    Vector3D center;
    double radius;
} Sphere;

int aabb_intersect(AABB *a, AABB *b) {
    return (a->min.x <= b->max.x && a->max.x >= b->min.x) &&
           (a->min.y <= b->max.y && a->max.y >= b->min.y) &&
           (a->min.z <= b->max.z && a->max.z >= b->min.z);
}

int sphere_intersect(Sphere *a, Sphere *b) {
    double dx = a->center.x - b->center.x;
    double dy = a->center.y - b->center.y;
    double dz = a->center.z - b->center.z;
    
    double dist_sq = dx * dx + dy * dy + dz * dz;
    double radius_sum = a->radius + b->radius;
    
    return dist_sq <= radius_sum * radius_sum;
}

int sphere_aabb_intersect(Sphere *s, AABB *box) {
    double closest_x = fmax(box->min.x, fmin(s->center.x, box->max.x));
    double closest_y = fmax(box->min.y, fmin(s->center.y, box->max.y));
    double closest_z = fmax(box->min.z, fmin(s->center.z, box->max.z));
    
    double dx = closest_x - s->center.x;
    double dy = closest_y - s->center.y;
    double dz = closest_z - s->center.z;
    
    double dist_sq = dx * dx + dy * dy + dz * dz;
    
    return dist_sq <= s->radius * s->radius;
}

int main() {
    Sphere *spheres = (Sphere*)malloc(NUM_OBJECTS * sizeof(Sphere));
    AABB *boxes = (AABB*)malloc(NUM_OBJECTS * sizeof(AABB));
    
    unsigned int seed = 42;
    for (int i = 0; i < NUM_OBJECTS; i++) {
        seed = seed * 1103515245 + 12345;
        spheres[i].center.x = ((seed & 0xFFFF) / (double)0xFFFF) * 100.0;
        seed = seed * 1103515245 + 12345;
        spheres[i].center.y = ((seed & 0xFFFF) / (double)0xFFFF) * 100.0;
        seed = seed * 1103515245 + 12345;
        spheres[i].center.z = ((seed & 0xFFFF) / (double)0xFFFF) * 100.0;
        seed = seed * 1103515245 + 12345;
        spheres[i].radius = ((seed & 0xFF) / (double)0xFF) * 5.0 + 1.0;
        
        boxes[i].min.x = spheres[i].center.x - 2.0;
        boxes[i].min.y = spheres[i].center.y - 2.0;
        boxes[i].min.z = spheres[i].center.z - 2.0;
        boxes[i].max.x = spheres[i].center.x + 2.0;
        boxes[i].max.y = spheres[i].center.y + 2.0;
        boxes[i].max.z = spheres[i].center.z + 2.0;
    }
    
    clock_t start = clock();
    
    int sphere_collisions = 0;
    int aabb_collisions = 0;
    int hybrid_collisions = 0;
    
    for (int i = 0; i < NUM_OBJECTS; i++) {
        for (int j = i + 1; j < NUM_OBJECTS; j++) {
            if (sphere_intersect(&spheres[i], &spheres[j])) {
                sphere_collisions++;
            }
            
            if (aabb_intersect(&boxes[i], &boxes[j])) {
                aabb_collisions++;
            }
            
            if (sphere_aabb_intersect(&spheres[i], &boxes[j])) {
                hybrid_collisions++;
            }
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Collision detection: %d objects, %.6f seconds\n", NUM_OBJECTS, time_spent);
    printf("Sphere: %d, AABB: %d, Hybrid: %d\n", 
           sphere_collisions, aabb_collisions, hybrid_collisions);
    
    free(spheres);
    free(boxes);
    
    return 0;
}
