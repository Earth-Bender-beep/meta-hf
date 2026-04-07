#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

typedef struct Stack {
    int* arr;
    int top;
} Stack;

Stack* createStack(int size) {
    Stack* stack = (Stack*)malloc(sizeof(Stack));
    stack->arr = (int*)malloc(size * sizeof(int));
    stack->top = -1;
    return stack;
}

void push(Stack* stack, int v) {
    stack->arr[++stack->top] = v;
}

int pop(Stack* stack) {
    return stack->arr[stack->top--];
}

int isEmpty(Stack* stack) {
    return stack->top == -1;
}

void DFSUtil(int v, int visited[], Node** adj) {
    visited[v] = 1;
    Node* temp = adj[v];
    
    while (temp) {
        if (!visited[temp->vertex])
            DFSUtil(temp->vertex, visited, adj);
        temp = temp->next;
    }
}

void fillOrder(int v, int visited[], Stack* stack, Node** adj) {
    visited[v] = 1;
    Node* temp = adj[v];
    
    while (temp) {
        if (!visited[temp->vertex])
            fillOrder(temp->vertex, visited, stack, adj);
        temp = temp->next;
    }
    
    push(stack, v);
}

int countSCCs(Node** adj, Node** transpose, int V) {
    Stack* stack = createStack(V);
    int* visited = (int*)calloc(V, sizeof(int));
    
    for (int i = 0; i < V; i++)
        if (!visited[i])
            fillOrder(i, visited, stack, adj);
    
    for (int i = 0; i < V; i++)
        visited[i] = 0;
    
    int count = 0;
    while (!isEmpty(stack)) {
        int v = pop(stack);
        if (!visited[v]) {
            DFSUtil(v, visited, transpose);
            count++;
        }
    }
    
    free(visited);
    free(stack->arr);
    free(stack);
    return count;
}

int main() {
    int V = 5000;
    Node** adj = (Node**)malloc(V * sizeof(Node*));
    Node** transpose = (Node**)malloc(V * sizeof(Node*));
    
    for (int i = 0; i < V; i++) {
        adj[i] = NULL;
        transpose[i] = NULL;
    }
    
    srand(42);
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < 3; j++) {
            int dest = rand() % V;
            
            Node* newNode = (Node*)malloc(sizeof(Node));
            newNode->vertex = dest;
            newNode->next = adj[i];
            adj[i] = newNode;
            
            Node* transNode = (Node*)malloc(sizeof(Node));
            transNode->vertex = i;
            transNode->next = transpose[dest];
            transpose[dest] = transNode;
        }
    }
    
    clock_t start = clock();
    int sccs = countSCCs(adj, transpose, V);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("SCCs: %d components in %d vertices, %.6f seconds\n", sccs, V, time_spent);
    
    for (int i = 0; i < V; i++) {
        Node* temp = adj[i];
        while (temp) {
            Node* toFree = temp;
            temp = temp->next;
            free(toFree);
        }
        temp = transpose[i];
        while (temp) {
            Node* toFree = temp;
            temp = temp->next;
            free(toFree);
        }
    }
    free(adj);
    free(transpose);
    
    return 0;
}
