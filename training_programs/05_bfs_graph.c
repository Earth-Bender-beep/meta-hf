#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

typedef struct Queue {
    int* items;
    int front;
    int rear;
    int size;
} Queue;

Queue* createQueue(int size) {
    Queue* q = (Queue*)malloc(sizeof(Queue));
    q->items = (int*)malloc(size * sizeof(int));
    q->front = -1;
    q->rear = -1;
    q->size = size;
    return q;
}

void enqueue(Queue* q, int value) {
    if (q->rear == q->size - 1)
        return;
    if (q->front == -1)
        q->front = 0;
    q->rear++;
    q->items[q->rear] = value;
}

int dequeue(Queue* q) {
    if (q->front == -1)
        return -1;
    int item = q->items[q->front];
    q->front++;
    if (q->front > q->rear)
        q->front = q->rear = -1;
    return item;
}

int isEmpty(Queue* q) {
    return q->front == -1;
}

void BFS(Node** adjList, int vertices, int start, int* visited) {
    Queue* q = createQueue(vertices);
    visited[start] = 1;
    enqueue(q, start);
    
    while (!isEmpty(q)) {
        int current = dequeue(q);
        Node* temp = adjList[current];
        
        while (temp) {
            int adjVertex = temp->vertex;
            if (!visited[adjVertex]) {
                visited[adjVertex] = 1;
                enqueue(q, adjVertex);
            }
            temp = temp->next;
        }
    }
    
    free(q->items);
    free(q);
}

int main() {
    int vertices = 8000;
    Node** adjList = (Node**)malloc(vertices * sizeof(Node*));
    int* visited = (int*)calloc(vertices, sizeof(int));
    
    for (int i = 0; i < vertices; i++)
        adjList[i] = NULL;
    
    srand(42);
    for (int i = 0; i < vertices; i++) {
        for (int j = 0; j < 5; j++) {
            int dest = rand() % vertices;
            Node* newNode = (Node*)malloc(sizeof(Node));
            newNode->vertex = dest;
            newNode->next = adjList[i];
            adjList[i] = newNode;
        }
    }
    
    clock_t start = clock();
    BFS(adjList, vertices, 0, visited);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("BFS: %d vertices in %.6f seconds\n", vertices, time_spent);
    
    for (int i = 0; i < vertices; i++) {
        Node* temp = adjList[i];
        while (temp) {
            Node* toFree = temp;
            temp = temp->next;
            free(toFree);
        }
    }
    free(adjList);
    free(visited);
    
    return 0;
}
