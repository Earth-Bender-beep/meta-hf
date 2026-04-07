// 2-SAT solver using strongly connected components (Kosaraju's algorithm)
// Determines satisfiability of 2-CNF boolean formulas
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_VARS 1000
#define MAX_CLAUSES 5000

typedef struct {
    int **adj;
    int **rev_adj;
    int *adj_count;
    int *rev_adj_count;
    int num_nodes;
} Graph;

typedef struct {
    int *stack;
    int top;
} Stack;

void push(Stack *s, int val) {
    s->stack[++s->top] = val;
}

int pop(Stack *s) {
    return s->stack[s->top--];
}

Graph* create_graph(int num_vars) {
    int num_nodes = 2 * num_vars;  // Each variable has positive and negative literals
    Graph *g = (Graph*)malloc(sizeof(Graph));
    
    g->adj = (int**)malloc(num_nodes * sizeof(int*));
    g->rev_adj = (int**)malloc(num_nodes * sizeof(int*));
    g->adj_count = (int*)calloc(num_nodes, sizeof(int));
    g->rev_adj_count = (int*)calloc(num_nodes, sizeof(int));
    
    for (int i = 0; i < num_nodes; i++) {
        g->adj[i] = (int*)malloc(MAX_CLAUSES * sizeof(int));
        g->rev_adj[i] = (int*)malloc(MAX_CLAUSES * sizeof(int));
    }
    
    g->num_nodes = num_nodes;
    return g;
}

void add_edge(Graph *g, int u, int v) {
    g->adj[u][g->adj_count[u]++] = v;
    g->rev_adj[v][g->rev_adj_count[v]++] = u;
}

void dfs1(Graph *g, int v, int *visited, Stack *finish_stack) {
    visited[v] = 1;
    
    for (int i = 0; i < g->adj_count[v]; i++) {
        int u = g->adj[v][i];
        if (!visited[u]) {
            dfs1(g, u, visited, finish_stack);
        }
    }
    
    push(finish_stack, v);
}

void dfs2(Graph *g, int v, int *visited, int *scc_id, int scc_num) {
    visited[v] = 1;
    scc_id[v] = scc_num;
    
    for (int i = 0; i < g->rev_adj_count[v]; i++) {
        int u = g->rev_adj[v][i];
        if (!visited[u]) {
            dfs2(g, u, visited, scc_id, scc_num);
        }
    }
}

int solve_2sat(Graph *g, int num_vars) {
    int *visited = (int*)calloc(g->num_nodes, sizeof(int));
    Stack finish_stack;
    finish_stack.stack = (int*)malloc(g->num_nodes * sizeof(int));
    finish_stack.top = -1;
    
    // First DFS on original graph
    for (int i = 0; i < g->num_nodes; i++) {
        if (!visited[i]) {
            dfs1(g, i, visited, &finish_stack);
        }
    }
    
    // Second DFS on reversed graph
    memset(visited, 0, g->num_nodes * sizeof(int));
    int *scc_id = (int*)malloc(g->num_nodes * sizeof(int));
    int scc_num = 0;
    
    while (finish_stack.top >= 0) {
        int v = pop(&finish_stack);
        if (!visited[v]) {
            dfs2(g, v, visited, scc_id, scc_num);
            scc_num++;
        }
    }
    
    // Check if x and ~x are in same SCC
    int satisfiable = 1;
    for (int i = 0; i < num_vars; i++) {
        if (scc_id[2*i] == scc_id[2*i + 1]) {
            satisfiable = 0;
            break;
        }
    }
    
    free(visited);
    free(scc_id);
    free(finish_stack.stack);
    
    return satisfiable;
}

int main() {
    int num_vars = 200;
    int num_clauses = 500;
    
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    int satisfiable_count = 0;
    int total_tests = 100;
    
    for (int test = 0; test < total_tests; test++) {
        Graph *g = create_graph(num_vars);
        
        // Generate random 2-SAT clauses
        for (int i = 0; i < num_clauses; i++) {
            seed = seed * 1103515245 + 12345;
            int var1 = seed % num_vars;
            seed = seed * 1103515245 + 12345;
            int var2 = seed % num_vars;
            seed = seed * 1103515245 + 12345;
            int neg1 = seed % 2;
            seed = seed * 1103515245 + 12345;
            int neg2 = seed % 2;
            
            // Clause: (lit1 OR lit2)
            // Implications: ~lit1 => lit2, ~lit2 => lit1
            int lit1 = 2 * var1 + neg1;
            int lit2 = 2 * var2 + neg2;
            int not_lit1 = 2 * var1 + (1 - neg1);
            int not_lit2 = 2 * var2 + (1 - neg2);
            
            add_edge(g, not_lit1, lit2);
            add_edge(g, not_lit2, lit1);
        }
        
        if (solve_2sat(g, num_vars)) {
            satisfiable_count++;
        }
        
        // Free graph
        for (int i = 0; i < g->num_nodes; i++) {
            free(g->adj[i]);
            free(g->rev_adj[i]);
        }
        free(g->adj);
        free(g->rev_adj);
        free(g->adj_count);
        free(g->rev_adj_count);
        free(g);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("2-SAT solver: vars=%d, clauses=%d, %d tests, %.6f seconds\n",
           num_vars, num_clauses, total_tests, time_spent);
    printf("Satisfiable: %d/%d\n", satisfiable_count, total_tests);
    
    return 0;
}
