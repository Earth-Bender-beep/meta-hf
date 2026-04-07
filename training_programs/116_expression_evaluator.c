// Expression evaluator using Shunting Yard algorithm and stack-based evaluation
// Parses and evaluates infix mathematical expressions with operators and parentheses
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

#define MAX_EXPR_LEN 256
#define NUM_TESTS 50000

typedef struct {
    double data[MAX_EXPR_LEN];
    int top;
} DoubleStack;

typedef struct {
    char data[MAX_EXPR_LEN];
    int top;
} CharStack;

void double_push(DoubleStack *s, double val) {
    if (s->top < MAX_EXPR_LEN) s->data[s->top++] = val;
}

double double_pop(DoubleStack *s) {
    return s->top > 0 ? s->data[--s->top] : 0;
}

void char_push(CharStack *s, char val) {
    if (s->top < MAX_EXPR_LEN) s->data[s->top++] = val;
}

char char_pop(CharStack *s) {
    return s->top > 0 ? s->data[--s->top] : '\0';
}

char char_peek(CharStack *s) {
    return s->top > 0 ? s->data[s->top - 1] : '\0';
}

int precedence(char op) {
    if (op == '+' || op == '-') return 1;
    if (op == '*' || op == '/') return 2;
    return 0;
}

int is_operator(char c) {
    return c == '+' || c == '-' || c == '*' || c == '/';
}

double apply_op(double a, double b, char op) {
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/': return b != 0 ? a / b : 0;
    }
    return 0;
}

double evaluate_expression(const char *expr) {
    DoubleStack values = {.top = 0};
    CharStack ops = {.top = 0};
    
    int i = 0;
    while (expr[i]) {
        if (isspace(expr[i])) {
            i++;
            continue;
        }
        
        if (isdigit(expr[i]) || expr[i] == '.') {
            double val = 0;
            int decimal = 0;
            double divisor = 1;
            
            while (isdigit(expr[i]) || expr[i] == '.') {
                if (expr[i] == '.') {
                    decimal = 1;
                } else if (decimal) {
                    divisor *= 10;
                    val += (expr[i] - '0') / divisor;
                } else {
                    val = val * 10 + (expr[i] - '0');
                }
                i++;
            }
            
            double_push(&values, val);
        } else if (expr[i] == '(') {
            char_push(&ops, expr[i]);
            i++;
        } else if (expr[i] == ')') {
            while (ops.top > 0 && char_peek(&ops) != '(') {
                char op = char_pop(&ops);
                double b = double_pop(&values);
                double a = double_pop(&values);
                double_push(&values, apply_op(a, b, op));
            }
            if (ops.top > 0) char_pop(&ops);  // Pop '('
            i++;
        } else if (is_operator(expr[i])) {
            while (ops.top > 0 && precedence(char_peek(&ops)) >= precedence(expr[i])) {
                char op = char_pop(&ops);
                double b = double_pop(&values);
                double a = double_pop(&values);
                double_push(&values, apply_op(a, b, op));
            }
            char_push(&ops, expr[i]);
            i++;
        } else {
            i++;
        }
    }
    
    while (ops.top > 0) {
        char op = char_pop(&ops);
        double b = double_pop(&values);
        double a = double_pop(&values);
        double_push(&values, apply_op(a, b, op));
    }
    
    return values.top > 0 ? values.data[0] : 0;
}

int main() {
    const char *expressions[] = {
        "3 + 5 * 2",
        "(3 + 5) * 2",
        "10 / 2 + 3",
        "2 * (3 + 4) - 5",
        "100 / (2 + 3) * 4"
    };
    int num_expr = 5;
    
    clock_t start = clock();
    
    double sum = 0.0;
    for (int iter = 0; iter < NUM_TESTS; iter++) {
        const char *expr = expressions[iter % num_expr];
        sum += evaluate_expression(expr);
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Expression evaluator: %d evaluations, %.6f seconds\n",
           NUM_TESTS, time_spent);
    printf("Sum of results: %.2f\n", sum);
    
    return 0;
}
