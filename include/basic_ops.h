#ifndef BASIC_OPS_H
#define BASIC_OPS_H

#include "autograd.h"

GraphNode* add_node(GraphNode* left, GraphNode* right);
GraphNode* sub_node(GraphNode* left, GraphNode* right);
GraphNode* mul_node(GraphNode* left, GraphNode* right);
GraphNode* div_node(GraphNode* left, GraphNode* right);
GraphNode* pow_node(GraphNode* base, double exponent);
GraphNode* log_node(GraphNode* input);
GraphNode* exp_node(GraphNode* input);

#endif