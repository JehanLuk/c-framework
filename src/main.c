#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "autograd.h"
#include "graph.h"
#include "tensor.h"
#include "errorcheck.h"
#include "basic_ops.h"

// Auxiliar function for errors
GraphNode* graph_error(MLInCERROR err) {
        mlinc_errno = err;
        return NULL;
}

GraphNode* node(double value) {
        GraphNode* n = malloc(sizeof(GraphNode));

        if (!n)
                graph_error(MLINC_OUT_OF_MEMORY_ERROR);
        
        n->extra.ndim = 0;
        n->extra.shape = NULL;
        n->extra.size = 0;
        n->extra.data = NULL;
        
        n->value = tensor_scalar(value);
        n->grad = tensor_scalar(0.0);
        n->right = NULL;
        n->left = NULL;
        n->backward = NULL;

        n->op = OP_LEAF;

        n->ref_count = 1;
        n->visited = 0;

        return n;
}

// Free memory functions (to avoid memory leak and preserve weights)

void retain(GraphNode* node) {
        if (node != NULL)
                node->ref_count++;
}

//TODO: Release is still susceptible to change
void release(GraphNode* node) {
        if (!node)
                return;

        if (node->ref_count <= 0)
                return;

        node->ref_count--;

        if (node->ref_count > 0)
                return;

        release(node->left);
        release(node->right);

        tensor_free(&node->value);
        tensor_free(&node->grad);
        tensor_free(&node->extra);

        node->left = NULL;
        node->right = NULL;
        node->backward = NULL;

        free(node);
}

//TODO: Reduction Operations

//TOPO (Topological sorting) and backward/backpropagation

void topo(GraphNode* n, GraphNode** list, int* size) {
        if (!list)
                graph_error(MLINC_NULL_POINTER_ERROR);

        if (!size)
                graph_error(MLINC_NULL_POINTER_ERROR);

        if (!n)
                return;

        if (n->visited)
                return;

        n->visited = 1;

        topo(n->left, list, size);
        topo(n->right, list, size);

        list[(*size)++] = n;
}

void backward(GraphNode* loss) {
        if (!loss)
                graph_error(MLINC_NULL_POINTER_ERROR);

        GraphNode* order [1000];
        int size = 0;

        topo(loss, order, &size);

        loss->grad.data[0] = 1.0;

        for (int i = size - 1; i >= 0; i--) {
                if(order[i]->backward) {
                        order[i]->backward(order[i]);
                }
        }

        for (int i = 0; i < size; i++) {
                order[i]->visited = 0;
        }
}

//Loss (using MSE (Mean Squared Error) as function) and optimization

GraphNode* mse(GraphNode* pred, GraphNode* target) {
        GraphNode* diff = sub_node(pred, target);
        return pow_node(diff, 2);
}

void step(GraphNode** params, int count, double lr) {
        if (!params)
                graph_error(MLINC_NULL_POINTER_ERROR);

        for (int i = 0; i < count; i++) {
                if (!params[i])
                        graph_error(MLINC_NULL_POINTER_ERROR);
                params[i]->value.data[0] -= lr * params[i]->grad.data[0];
                params[i]->grad.data[0] = 0.0;
        }
}

int main() {
        GraphNode* w = node(-3.0);
        GraphNode* b = node(10.0);

        GraphNode* x = node(2.0);
        GraphNode* target = node(12.0);

        for (int epoch = 0; epoch < 500; epoch++) {

                GraphNode* wx = mul_node(w, x);
                GraphNode* pred = add_node(wx, b);
                GraphNode* loss = mse(pred, target);

                backward(loss);

                GraphNode* params[] = {w,b};

                step(params,2,0.01);

                char filename[128];
                sprintf(filename,
                        "epochs/epoch_%03d.dot",
                        epoch);

                graph_export(loss, filename);

                printf(
                        "Epoch %d | Loss %.15f | Weight %.15f | Bias %.15f\n",
                        epoch,
                        loss->value.data[0],
                        w->value.data[0],
                        b->value.data[0]
                );

                release(loss);
        }

        release(w);
        release(b);
        release(x);
        release(target);

        return 0;
}