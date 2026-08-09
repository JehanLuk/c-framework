#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.h"

typedef enum {
    //Starting node
    OP_LEAF,

    //Basic operations
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_POW,
    OP_LOG,
    OP_EXP,

    //TODO: Matrix operations
    OP_MATMUL,
    OP_TRANS,
    OP_RESHAPE,
    OP_BROADCAST,

    //TODO: Reduction operations
    OP_SUM,
    OP_MEAN,
    OP_MAX,

    //TODO: Activation functions
    OP_RELU,
    OP_SIGMOID,
    OP_TAHN,
    OP_SOFTMAX
} Operation;

//Node struct definition and creation
typedef struct GraphNode {
    Tensor value;
    Tensor grad;

    Tensor extra;
    Operation op;

    int ref_count;
    
    int visited;

    struct GraphNode* right;
    struct GraphNode* left;

    void (*backward)(struct GraphNode*);
} GraphNode;

void topo(GraphNode* n, GraphNode** list, int* size);

#endif