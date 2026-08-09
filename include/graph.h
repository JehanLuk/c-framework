#ifndef GRAPH_H
#define GRAPH_H

#include "autograd.h"
#include "tensor.h"

void graph_export(GraphNode* root, const char* filename);

GraphNode* node(double value);

void retain(GraphNode* node);
void release(GraphNode* node);

void free_tensor(Tensor* t);

#endif