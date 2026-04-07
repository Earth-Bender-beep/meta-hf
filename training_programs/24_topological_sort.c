#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

typedef struct Stack {
    int top;
    int capacity;
    int* array;
} Stack;

Stack* createStack(int capacity) {
    Stack* stack = (Stack*)malloc(sizeof(Stack));
    stack->capacity = capacity;
    stack->top = -1;
    stack->array = (int*)malloc(capacity * sizeof(int));
    return stack;
}

void push(Stack* stack, int item) {
    stack->array[++stack->top] = item;
}

void topologicalSortUtil(int v, int visited[], Stack* stack, Node** adj) {
    visited[v] = 1;
    
    Node* temp = adj[v];
    while (temp) {
        if (!visited[temp->vertex])
            topologicalSortUtil(temp->vertex, visited, stack, adj);
        temp = temp->next;
    }
    
    push(stack, v);
}

void topologicalSort(Node** adj, int V) {
    Stack* stack = createStack(V);
    int* visited = (int*)calloc(V, sizeof(int));
    
    for (int i = 0; i < V; i++)
        if (visited[i] == 0)
            topologicalSortUtil(i, visited, stack, adj);
    
    free(visited);
    free(stack->array);
    free(stack);
}

int main() {
    int V = 10000;
    Node** adj = (Node**)malloc(V * sizeof(Node*));
    
    for (int i = 0; i < V; i++)
        adj[i] = NULL;
    
    srand(42);
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < 3; j++) {
            int dest = (i + rand() % (V - i)) % V;
            if (dest != i) {
                Node* newNode = (Node*)malloc(sizeof(Node));
                newNode->vertex = dest;
                newNode->next = adj[i];
                adj[i] = newNode;
            }
        }
    }
    
    clock_t start = clock();
    topologicalSort(adj, V);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Topological sort: %d vertices in %.6f seconds\n", V, time_spent);
    
    for (int i = 0; i < V; i++) {
        Node* temp = adj[i];
        while (temp) {
            Node* toFree = temp;
            temp = temp->next;
            free(toFree);
        }
    }
    free(adj);
    
    return 0;
}
