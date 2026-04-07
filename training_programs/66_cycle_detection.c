// Cycle detection in graphs and linked lists using Floyd's algorithm
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_NODES 1000

typedef struct Node {
    int value;
    struct Node *next;
} Node;

typedef struct {
    int adj[MAX_NODES][MAX_NODES];
    int num_nodes;
} Graph;

Node* create_node(int value) {
    Node *node = (Node*)malloc(sizeof(Node));
    node->value = value;
    node->next = NULL;
    return node;
}

int detect_cycle_floyd(Node *head) {
    if (!head) return 0;
    
    Node *slow = head;
    Node *fast = head;
    
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        
        if (slow == fast) {
            return 1;
        }
    }
    
    return 0;
}

int detect_cycle_graph_dfs(Graph *g, int node, int *visited, int *rec_stack) {
    visited[node] = 1;
    rec_stack[node] = 1;
    
    for (int i = 0; i < g->num_nodes; i++) {
        if (g->adj[node][i]) {
            if (!visited[i]) {
                if (detect_cycle_graph_dfs(g, i, visited, rec_stack)) {
                    return 1;
                }
            } else if (rec_stack[i]) {
                return 1;
            }
        }
    }
    
    rec_stack[node] = 0;
    return 0;
}

int main() {
    unsigned int seed = 42;
    
    clock_t start = clock();
    
    int cycles_found = 0;
    
    for (int test = 0; test < 100; test++) {
        Node *head = create_node(0);
        Node *current = head;
        Node *cycle_point = NULL;
        
        for (int i = 1; i < 1000; i++) {
            current->next = create_node(i);
            current = current->next;
            
            if (i == 500) cycle_point = current;
        }
        
        seed = seed * 1103515245 + 12345;
        if (seed % 2 == 0) {
            current->next = cycle_point;
        }
        
        if (detect_cycle_floyd(head)) {
            cycles_found++;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Cycle detection: 100 tests, %.6f seconds\n", time_spent);
    printf("Cycles found: %d\n", cycles_found);
    
    return 0;
}
