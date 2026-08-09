#include <stdio.h>
#include "graph.h"

extern void topo(GraphNode* n, GraphNode** list, int* size);

const char* op_name(Operation op) {
    switch (op) {
        case OP_LEAF:
            return "LEAF";
        case OP_ADD:
            return "ADD";
        case OP_SUB:
            return "SUB";
        case OP_MUL:
            return "MUL";
        case OP_DIV:
            return "DIV";
        case OP_POW:
            return "POW";
        case OP_EXP:
            return "EXP";
        case OP_LOG:
            return "LOG";
        default:
            return "UNKNOWN";
    }
}

void graph_export(GraphNode* root, const char* filename) {
    FILE* f = fopen(filename, "w");

    if (!f) {
        printf("Error creating %s\n", filename);
        return;
    }

    fprintf(f, "digraph G {\n");
    fprintf(f, "    rankdir=TB;\n");
    fprintf(f, "    node [shape=record];\n\n");
    
    GraphNode* order[1000];
    int size = 0;
    topo(root, order, &size);

    for (int i = 0; i < size; i++) {
        GraphNode* n = order[i];

        if (n->op == OP_LEAF) {
            fprintf(f, "N%p [shape=circle,label=\"%.4f\"];\n", (void*)n, n->value.data[0]);
        }
        else {
            fprintf(f, "N%p [shape=box,label=\"%s\"];\n", (void*)n, op_name(n->op));
        }
    }
    fprintf(f, "\n");

    for (int i = 0; i < size; i++) {
        GraphNode* n = order[i];
        if (n->left)
            fprintf(f, "N%p -> N%p;\n", (void*)n->left, (void*)n);
        if (n->right)
            fprintf(f, "N%p -> N%p;\n", (void*)n->right, (void*)n);
    }
    fprintf(f, "}\n");

    fclose(f);
}