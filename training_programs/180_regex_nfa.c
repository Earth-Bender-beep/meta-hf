// Regular expression matching using NFA (Thompson's construction)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_STATES 1000
#define MAX_PATTERN 100

typedef struct State {
    int is_end;
    char c;
    struct State *out1;
    struct State *out2;
} State;

typedef struct {
    State *start;
    State *end;
} Fragment;

State* create_state(char c, State *out1, State *out2) {
    State *s = (State*)malloc(sizeof(State));
    s->is_end = 0;
    s->c = c;
    s->out1 = out1;
    s->out2 = out2;
    return s;
}

Fragment create_fragment(State *start, State *end) {
    Fragment f;
    f.start = start;
    f.end = end;
    return f;
}

Fragment compile_char(char c) {
    State *s = create_state(c, NULL, NULL);
    return create_fragment(s, s);
}

Fragment compile_concat(Fragment f1, Fragment f2) {
    f1.end->out1 = f2.start;
    return create_fragment(f1.start, f2.end);
}

Fragment compile_alternate(Fragment f1, Fragment f2) {
    State *s = create_state(0, f1.start, f2.start);
    State *e = create_state(0, NULL, NULL);
    f1.end->out1 = e;
    f2.end->out1 = e;
    return create_fragment(s, e);
}

Fragment compile_star(Fragment f) {
    State *s = create_state(0, f.start, NULL);
    f.end->out1 = s;
    return create_fragment(s, s);
}

void add_state(State **list, int *count, State *s) {
    if (s == NULL || *count >= MAX_STATES) return;
    
    for (int i = 0; i < *count; i++) {
        if (list[i] == s) return;
    }
    
    list[(*count)++] = s;
    
    if (s->c == 0) {
        add_state(list, count, s->out1);
        add_state(list, count, s->out2);
    }
}

int match(State *start, const char *text) {
    State *current_states[MAX_STATES];
    State *next_states[MAX_STATES];
    int current_count = 0;
    int next_count = 0;
    
    add_state(current_states, &current_count, start);
    
    for (int i = 0; text[i]; i++) {
        next_count = 0;
        
        for (int j = 0; j < current_count; j++) {
            State *s = current_states[j];
            if (s->c == text[i] || s->c == '.') {
                add_state(next_states, &next_count, s->out1);
            }
        }
        
        State **temp = current_states;
        current_states = next_states;
        next_states = temp;
        current_count = next_count;
    }
    
    for (int i = 0; i < current_count; i++) {
        if (current_states[i]->is_end) {
            return 1;
        }
    }
    
    return 0;
}

Fragment compile_simple_regex(const char *pattern) {
    Fragment stack[MAX_PATTERN];
    int stack_top = -1;
    
    for (int i = 0; pattern[i]; i++) {
        if (pattern[i] == '|') {
            Fragment f2 = stack[stack_top--];
            Fragment f1 = stack[stack_top--];
            stack[++stack_top] = compile_alternate(f1, f2);
        } else if (pattern[i] == '*' && stack_top >= 0) {
            Fragment f = stack[stack_top--];
            stack[++stack_top] = compile_star(f);
        } else if (pattern[i] == '.') {
            stack[++stack_top] = compile_char('.');
        } else {
            stack[++stack_top] = compile_char(pattern[i]);
        }
        
        // Auto-concat adjacent characters
        while (stack_top > 0 && pattern[i + 1] && pattern[i + 1] != '|' && pattern[i + 1] != '*') {
            i++;
            Fragment f2 = compile_char(pattern[i]);
            Fragment f1 = stack[stack_top--];
            stack[++stack_top] = compile_concat(f1, f2);
        }
    }
    
    if (stack_top >= 0) {
        stack[stack_top].end->is_end = 1;
        return stack[stack_top];
    }
    
    State *dummy = create_state(0, NULL, NULL);
    return create_fragment(dummy, dummy);
}

int main() {
    const char *patterns[] = {
        "abc",
        "a.c",
        "ab*c",
        "a|b"
    };
    
    const char *texts[] = {
        "abc",
        "adc",
        "abbbbc",
        "a",
        "b",
        "xyz"
    };
    
    int num_patterns = 4;
    int num_texts = 6;
    
    clock_t start = clock();
    
    int total_matches = 0;
    for (int test = 0; test < 1000; test++) {
        for (int p = 0; p < num_patterns; p++) {
            Fragment nfa = compile_simple_regex(patterns[p]);
            
            for (int t = 0; t < num_texts; t++) {
                if (match(nfa.start, texts[t])) {
                    total_matches++;
                }
            }
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Regex NFA: %d patterns x %d texts x 1000 iterations, %.6f seconds\n",
           num_patterns, num_texts, time_spent);
    printf("Total matches: %d\n", total_matches);
    
    return 0;
}
