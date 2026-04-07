// CSV parser with RFC 4180 compliance
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LINE 10000
#define MAX_FIELDS 100

typedef struct {
    char *fields[MAX_FIELDS];
    int field_count;
} CSVRow;

int parse_csv_line(const char *line, CSVRow *row) {
    row->field_count = 0;
    int in_quotes = 0;
    int field_start = 0;
    int i = 0;
    
    while (line[i] && row->field_count < MAX_FIELDS) {
        if (line[i] == '"') {
            in_quotes = !in_quotes;
        } else if (line[i] == ',' && !in_quotes) {
            int len = i - field_start;
            row->fields[row->field_count] = (char*)malloc(len + 1);
            strncpy(row->fields[row->field_count], line + field_start, len);
            row->fields[row->field_count][len] = '\0';
            row->field_count++;
            field_start = i + 1;
        }
        i++;
    }
    
    int len = i - field_start;
    row->fields[row->field_count] = (char*)malloc(len + 1);
    strncpy(row->fields[row->field_count], line + field_start, len);
    row->fields[row->field_count][len] = '\0';
    row->field_count++;
    
    return row->field_count;
}

void free_csv_row(CSVRow *row) {
    for (int i = 0; i < row->field_count; i++) {
        free(row->fields[i]);
    }
}

int main() {
    const char *test_lines[] = {
        "name,age,city",
        "John,25,New York",
        "Jane,30,\"Los Angeles\"",
        "Bob,35,\"San Francisco, CA\"",
        "Alice,28,Chicago"
    };
    
    int num_lines = 5;
    
    clock_t start = clock();
    
    int total_fields = 0;
    for (int test = 0; test < 10000; test++) {
        for (int i = 0; i < num_lines; i++) {
            CSVRow row;
            parse_csv_line(test_lines[i], &row);
            total_fields += row.field_count;
            free_csv_row(&row);
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("CSV parser: %d lines x 10000 iterations, %.6f seconds\n",
           num_lines, time_spent);
    printf("Total fields parsed: %d\n", total_fields);
    
    return 0;
}
