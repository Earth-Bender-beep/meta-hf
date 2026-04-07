// Tarjan's algorithm for finding strongly connected components in directed graphs
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_NODES 1000
#define MIN(a,b) ((a) < (b) ? (a) : (b))

typedef struct {
    int **adj_matrix;
    int num_nodes;
} Graph;

typedef struct {
    int *data;
    int top;
} Stack;

Stack* create_stack(int capacity) {
    Stack *stack = (Stack*)malloc(sizeof(Stack));
    stack->data = (int*)malloc(capacity * sizeof(int));
    stack->top = -1;
    return stack;
}

void push(Stack *stack, int value) {
    stack->data[++stack->top] = value;
}

int pop(Stack *stack) {
    return stack->data[stack->top--];
}

int is_empty(Stack *stack) {
    return stack->top == -1;
}

void tarjan_scc_util(Graph *g, int u, int *disc, int *low, Stack *stack, 
                     int *on_stack, int *time, int *scc_count, int **sccs, int *scc_sizes) {
    disc[u] = low[u] = ++(*time);
    push(stack, u);
    on_stack[u] = 1;
    
    for (int v = 0; v < g->num_nodes; v++) {
        if (g->adj_matrix[u][v]) {
            if (disc[v] == -1) {
                tarjan_scc_util(g, v, disc, low, stack, on_stack, time, scc_count, sccs, scc_sizes);
                low[u] = MIN(low[u], low[v]);
            } else if (on_stack[v]) {
                low[u] = MIN(low[u], disc[v]);
            }
        }
    }
    
    if (low[u] == disc[u]) {
        int w;
        int scc_id = *scc_count;
        do {
            w = pop(stack);
            on_stack[w] = 0;
            sccs[scc_id][scc_sizes[scc_id]++] = w;
        } while (w != u);
        (*scc_count)++;
    }
}

int tarjan_scc(Graph *g, int **sccs, int *scc_sizes) {
    int *disc = (int*)malloc(g->num_nodes * sizeof(int));
    int *low = (int*)malloc(g->num_nodes * sizeof(int));
    int *on_stack = (int*)calloc(g->num_nodes, sizeof(int));
    
    for (int i = 0; i < g->num_nodes; i++) {
        disc[i] = -1;
        low[i] = -1;
    }
    
    Stack *stack = create_stack(g->num_nodes);
    int time = 0;
    int scc_count = 0;
    
    for (int i = 0; i < g->num_nodes; i++) {
        if (disc[i] == -1) {
            tarjan_scc_util(g, i, disc, low, stack, on_stack, &time, &scc_count, sccs, scc_sizes);
        }
    }
    
    free(disc);
    free(low);
    free(on_stack);
    free(stack->data);
    free(stack);
    
    return scc_count;
}

Graph* create_graph(int n, unsigned int *seed) {
    Graph *g = (Graph*)malloc(sizeof(Graph));
    g->num_nodes = n;
    g->adj_matrix = (int**)malloc(n * sizeof(int*));
    
    for (int i = 0; i < n; i++) {
        g->adj_matrix[i] = (int*)calloc(n, sizeof(int));
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                *seed = *seed * 1103515245 + 12345;
                if ((*seed & 0xFF) < 30) {
                    g->adj_matrix[i][j] = 1;
                }
            }
        }
    }
    
    return g;
}

int main() {
    int n = 100;
    unsigned int seed = 42;
    
    Graph *g = create_graph(n, &seed);
    
    int **sccs = (int**)malloc(n * sizeof(int*));
    int *scc_sizes = (int*)calloc(n, sizeof(int));
    
    for (int i = 0; i < n; i++) {
        sccs[i] = (int*)malloc(n * sizeof(int));
    }
    
    clock_t start = clock();
    int num_sccs = tarjan_scc(g, sccs, scc_sizes);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    int largest_scc = 0;
    for (int i = 0; i < num_sccs; i++) {
        if (scc_sizes[i] > largest_scc) {
            largest_scc = scc_sizes[i];
        }
    }
    
    printf("Tarjan SCC: %d nodes, %d SCCs found, largest size=%d, %.6f seconds\n",
           n, num_sccs, largest_scc, time_spent);
    
    for (int i = 0; i < n; i++) {
        free(g->adj_matrix[i]);
        free(sccs[i]);
    }
    free(g->adj_matrix);
    free(g);
    free(sccs);
    free(scc_sizes);
    
    return 0;
}
