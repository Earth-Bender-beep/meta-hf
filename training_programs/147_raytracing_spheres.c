// Simple ray tracing with sphere intersection
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define IMAGE_WIDTH 256
#define IMAGE_HEIGHT 256
#define NUM_SPHERES 10

typedef struct {
    double x, y, z;
} Vec3;

typedef struct {
    Vec3 center;
    double radius;
    double color[3];
} Sphere;

typedef struct {
    Vec3 origin;
    Vec3 direction;
} Ray;

double dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 sub(Vec3 a, Vec3 b) {
    Vec3 result = {a.x - b.x, a.y - b.y, a.z - b.z};
    return result;
}

int intersect_sphere(Ray *ray, Sphere *sphere, double *t) {
    Vec3 oc = sub(ray->origin, sphere->center);
    
    double a = dot(ray->direction, ray->direction);
    double b = 2.0 * dot(oc, ray->direction);
    double c = dot(oc, oc) - sphere->radius * sphere->radius;
    
    double discriminant = b * b - 4 * a * c;
    
    if (discriminant < 0) {
        return 0;
    }
    
    *t = (-b - sqrt(discriminant)) / (2.0 * a);
    
    return (*t > 0.001);
}

void trace_ray(Ray *ray, Sphere *spheres, int num_spheres, double *color) {
    double min_t = 1e30;
    int hit_sphere = -1;
    double t;
    
    for (int i = 0; i < num_spheres; i++) {
        if (intersect_sphere(ray, &spheres[i], &t)) {
            if (t < min_t) {
                min_t = t;
                hit_sphere = i;
            }
        }
    }
    
    if (hit_sphere >= 0) {
        color[0] = spheres[hit_sphere].color[0];
        color[1] = spheres[hit_sphere].color[1];
        color[2] = spheres[hit_sphere].color[2];
    } else {
        color[0] = 0.5;
        color[1] = 0.7;
        color[2] = 1.0;
    }
}

int main() {
    Sphere *spheres = (Sphere*)malloc(NUM_SPHERES * sizeof(Sphere));
    double *image = (double*)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(double));
    
    unsigned int seed = 42;
    for (int i = 0; i < NUM_SPHERES; i++) {
        seed = seed * 1103515245 + 12345;
        spheres[i].center.x = ((seed & 0xFFFF) / (double)0xFFFF) * 10.0 - 5.0;
        seed = seed * 1103515245 + 12345;
        spheres[i].center.y = ((seed & 0xFFFF) / (double)0xFFFF) * 10.0 - 5.0;
        seed = seed * 1103515245 + 12345;
        spheres[i].center.z = -((seed & 0xFF) / (double)0xFF) * 5.0 - 5.0;
        seed = seed * 1103515245 + 12345;
        spheres[i].radius = ((seed & 0xFF) / (double)0xFF) * 1.5 + 0.5;
        
        seed = seed * 1103515245 + 12345;
        spheres[i].color[0] = ((seed & 0xFF) / (double)0xFF);
        seed = seed * 1103515245 + 12345;
        spheres[i].color[1] = ((seed & 0xFF) / (double)0xFF);
        seed = seed * 1103515245 + 12345;
        spheres[i].color[2] = ((seed & 0xFF) / (double)0xFF);
    }
    
    clock_t start = clock();
    
    Vec3 camera = {0, 0, 0};
    
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            Ray ray;
            ray.origin = camera;
            ray.direction.x = (x / (double)IMAGE_WIDTH - 0.5) * 2.0;
            ray.direction.y = (y / (double)IMAGE_HEIGHT - 0.5) * 2.0;
            ray.direction.z = -1.0;
            
            double color[3];
            trace_ray(&ray, spheres, NUM_SPHERES, color);
            
            int idx = (y * IMAGE_WIDTH + x) * 3;
            image[idx] = color[0];
            image[idx + 1] = color[1];
            image[idx + 2] = color[2];
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Ray tracing: %dx%d image, %d spheres, %.6f seconds\n",
           IMAGE_WIDTH, IMAGE_HEIGHT, NUM_SPHERES, time_spent);
    
    free(spheres);
    free(image);
    
    return 0;
}
