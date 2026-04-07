// Runge-Kutta 4th order ODE solver
// Numerical integration of differential equations
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_STEPS 10000
#define DT 0.01

// dy/dt = -2y + x
double ode1(double t, double y) {
    return -2.0 * y + t;
}

// Lorenz system component
typedef struct {
    double x, y, z;
} Vector3;

void lorenz_system(double t, Vector3 state, Vector3 *deriv) {
    double sigma = 10.0;
    double rho = 28.0;
    double beta = 8.0/3.0;
    
    deriv->x = sigma * (state.y - state.x);
    deriv->y = state.x * (rho - state.z) - state.y;
    deriv->z = state.x * state.y - beta * state.z;
}

double rk4_step(double (*f)(double, double), double t, double y, double dt) {
    double k1 = f(t, y);
    double k2 = f(t + dt/2.0, y + dt*k1/2.0);
    double k3 = f(t + dt/2.0, y + dt*k2/2.0);
    double k4 = f(t + dt, y + dt*k3);
    
    return y + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
}

void rk4_step_vector(Vector3 state, Vector3 *next, double t, double dt) {
    Vector3 k1, k2, k3, k4;
    Vector3 temp;
    
    lorenz_system(t, state, &k1);
    
    temp.x = state.x + dt*k1.x/2.0;
    temp.y = state.y + dt*k1.y/2.0;
    temp.z = state.z + dt*k1.z/2.0;
    lorenz_system(t + dt/2.0, temp, &k2);
    
    temp.x = state.x + dt*k2.x/2.0;
    temp.y = state.y + dt*k2.y/2.0;
    temp.z = state.z + dt*k2.z/2.0;
    lorenz_system(t + dt/2.0, temp, &k3);
    
    temp.x = state.x + dt*k3.x;
    temp.y = state.y + dt*k3.y;
    temp.z = state.z + dt*k3.z;
    lorenz_system(t + dt, temp, &k4);
    
    next->x = state.x + (dt/6.0) * (k1.x + 2.0*k2.x + 2.0*k3.x + k4.x);
    next->y = state.y + (dt/6.0) * (k1.y + 2.0*k2.y + 2.0*k3.y + k4.y);
    next->z = state.z + (dt/6.0) * (k1.z + 2.0*k2.z + 2.0*k3.z + k4.z);
}

int main() {
    clock_t start = clock();
    
    // Solve simple ODE
    double t = 0.0;
    double y = 1.0;
    
    for (int i = 0; i < N_STEPS; i++) {
        y = rk4_step(ode1, t, y, DT);
        t += DT;
    }
    
    // Solve Lorenz system
    Vector3 state = {1.0, 1.0, 1.0};
    double t_lorenz = 0.0;
    
    for (int i = 0; i < N_STEPS; i++) {
        Vector3 next;
        rk4_step_vector(state, &next, t_lorenz, DT);
        state = next;
        t_lorenz += DT;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("RK4 ODE solver (%d steps): %.6f seconds\n", N_STEPS, time_spent);
    printf("Simple ODE final: y(%.2f) = %.6f\n", t, y);
    printf("Lorenz final: (%.6f, %.6f, %.6f)\n", state.x, state.y, state.z);
    
    return 0;
}
