// Simple JSON parser - recursive descent parsing
// String processing, state machine, recursive structure
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

#define MAX_DEPTH 20
#define BUFFER_SIZE 5000

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} JsonType;

typedef struct {
    const char *json;
    int pos;
    int error;
} JsonParser;

void skip_whitespace(JsonParser *p) {
    while (p->json[p->pos] && isspace(p->json[p->pos])) {
        p->pos++;
    }
}

int parse_value(JsonParser *p, int depth);

int parse_string(JsonParser *p) {
    if (p->json[p->pos] != '"') return 0;
    p->pos++;
    
    while (p->json[p->pos] && p->json[p->pos] != '"') {
        if (p->json[p->pos] == '\\') {
            p->pos += 2;
        } else {
            p->pos++;
        }
    }
    
    if (p->json[p->pos] == '"') {
        p->pos++;
        return 1;
    }
    
    p->error = 1;
    return 0;
}

int parse_number(JsonParser *p) {
    int start = p->pos;
    
    if (p->json[p->pos] == '-') p->pos++;
    
    if (!isdigit(p->json[p->pos])) return 0;
    
    while (isdigit(p->json[p->pos])) p->pos++;
    
    if (p->json[p->pos] == '.') {
        p->pos++;
        while (isdigit(p->json[p->pos])) p->pos++;
    }
    
    if (p->json[p->pos] == 'e' || p->json[p->pos] == 'E') {
        p->pos++;
        if (p->json[p->pos] == '+' || p->json[p->pos] == '-') p->pos++;
        while (isdigit(p->json[p->pos])) p->pos++;
    }
    
    return p->pos > start;
}

int parse_array(JsonParser *p, int depth) {
    if (p->json[p->pos] != '[') return 0;
    p->pos++;
    
    skip_whitespace(p);
    
    if (p->json[p->pos] == ']') {
        p->pos++;
        return 1;
    }
    
    while (1) {
        if (!parse_value(p, depth + 1)) {
            p->error = 1;
            return 0;
        }
        
        skip_whitespace(p);
        
        if (p->json[p->pos] == ']') {
            p->pos++;
            return 1;
        }
        
        if (p->json[p->pos] == ',') {
            p->pos++;
            skip_whitespace(p);
        } else {
            p->error = 1;
            return 0;
        }
    }
}

int parse_object(JsonParser *p, int depth) {
    if (p->json[p->pos] != '{') return 0;
    p->pos++;
    
    skip_whitespace(p);
    
    if (p->json[p->pos] == '}') {
        p->pos++;
        return 1;
    }
    
    while (1) {
        skip_whitespace(p);
        
        if (!parse_string(p)) {
            p->error = 1;
            return 0;
        }
        
        skip_whitespace(p);
        
        if (p->json[p->pos] != ':') {
            p->error = 1;
            return 0;
        }
        p->pos++;
        
        if (!parse_value(p, depth + 1)) {
            p->error = 1;
            return 0;
        }
        
        skip_whitespace(p);
        
        if (p->json[p->pos] == '}') {
            p->pos++;
            return 1;
        }
        
        if (p->json[p->pos] == ',') {
            p->pos++;
        } else {
            p->error = 1;
            return 0;
        }
    }
}

int parse_value(JsonParser *p, int depth) {
    if (depth > MAX_DEPTH) {
        p->error = 1;
        return 0;
    }
    
    skip_whitespace(p);
    
    char c = p->json[p->pos];
    
    if (c == '"') return parse_string(p);
    if (c == '[') return parse_array(p, depth);
    if (c == '{') return parse_object(p, depth);
    if (c == '-' || isdigit(c)) return parse_number(p);
    
    if (strncmp(&p->json[p->pos], "true", 4) == 0) {
        p->pos += 4;
        return 1;
    }
    if (strncmp(&p->json[p->pos], "false", 5) == 0) {
        p->pos += 5;
        return 1;
    }
    if (strncmp(&p->json[p->pos], "null", 4) == 0) {
        p->pos += 4;
        return 1;
    }
    
    p->error = 1;
    return 0;
}

void generate_json(char *buffer, int size) {
    snprintf(buffer, size,
        "{\"name\":\"test\",\"value\":123.456,\"active\":true,"
        "\"items\":[1,2,3,4,5],\"nested\":{\"a\":1,\"b\":2},"
        "\"array\":[{\"x\":1},{\"x\":2},{\"x\":3}],"
        "\"long_array\":[%s]}",
        "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"
    );
}

int main() {
    char json_buffer[BUFFER_SIZE];
    generate_json(json_buffer, BUFFER_SIZE);
    
    clock_t start = clock();
    
    int valid_count = 0;
    for (int i = 0; i < 1000; i++) {
        JsonParser parser = {json_buffer, 0, 0};
        if (parse_value(&parser, 0) && !parser.error) {
            valid_count++;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("JSON parsing: 1000 iterations, %.6f seconds\n", time_spent);
    printf("Valid parses: %d, JSON length: %zu bytes\n", valid_count, strlen(json_buffer));
    
    return 0;
}
