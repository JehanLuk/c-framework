#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "graph.h"
#include "tensor.h"
#include "basic_ops.h"
#include "errorcheck.h"

int graph_validate_binary(GraphNode* left, GraphNode* right) {
        if (!left || !right) {
                mlinc_errno = MLINC_NULL_POINTER_ERROR;
                return 0;
        }

        return 1;
}

//Basic Operations

void backward_add(GraphNode* self) {
        double g = self->grad.data[0];

        self->left->grad.data[0] += g;
        self->right->grad.data[0] += g;
}

GraphNode* add_node(GraphNode* left, GraphNode* right) {
        if (!graph_validate_binary(left, right))
                return NULL;

        double vleft = left->value.data[0];
        double vright = right->value.data[0];

        GraphNode* out = node(vleft + vright);

        out->left = left;
        out->right = right;
        out->backward = backward_add;

        out->op = OP_ADD;

        retain(left);
        retain(right);

        return out;
}

void backward_sub(GraphNode* self) {
        double g = self->grad.data[0];

        self->left->grad.data[0] += 1.0 * g;
        self->right->grad.data[0] += -1.0 * g;
}

GraphNode* sub_node(GraphNode* left, GraphNode* right) {
        if (!graph_validate_binary(left, right))
                return NULL;

        double vleft = left->value.data[0];
        double vright = right->value.data[0];
        
        GraphNode* out = node(vleft - vright);

        out->left = left;
        out->right = right;
        out->backward = backward_sub;

        out->op = OP_SUB;

        retain(left);
        retain(right);

        return out;
}

void backward_mul(GraphNode* self) {
        double g = self->grad.data[0];
        
        self->left->grad.data[0] += self->right->value.data[0] * g;
        self->right->grad.data[0] += self->left->value.data[0] * g;
}

GraphNode* mul_node(GraphNode* left, GraphNode* right) {
        if (!graph_validate_binary(left, right))
                return NULL;

        double vleft = left->value.data[0];
        double vright = right->value.data[0];

        GraphNode* out = node(vleft * vright);

        out->left = left;
        out->right = right;
        out->backward = backward_mul;

        out->op = OP_MUL;

        retain(left);
        retain(right);

        return out;
}

void backward_div(GraphNode* self) {
        double g = self->grad.data[0];

        double numerator = self->left->value.data[0];
        double denominator = self->right->value.data[0];

        self->left->grad.data[0] += (1.0 / denominator) * g;
        self->right->grad.data[0] += (-numerator / (denominator * denominator)) * g;
}

GraphNode* div_node(GraphNode* left, GraphNode* right) {
        if (!graph_validate_binary(left, right))
                return NULL;

        double vnumerator = left->value.data[0];
        double vdenominator = right->value.data[0];

        if (vdenominator == 0.0)
                return graph_error(MLINC_DIVISION_BY_ZERO_ERROR);

        GraphNode* out = node(vnumerator / vdenominator);

        out->left = left;
        out->right = right;
        out->backward = backward_div;

        out->op = OP_DIV;

        retain(left);
        retain(right);

        return out;
}

void backward_pow(GraphNode* self) {
        double g = self->grad.data[0];

        double base = self->left->value.data[0];
        double exponent = self->extra.data[0];

        self->left->grad.data[0] += exponent * pow(base, exponent - 1.0) * g;
}

GraphNode* pow_node(GraphNode* base, double exponent) {
        if (!base)
                return graph_error(MLINC_NULL_POINTER_ERROR);

        double vbase = base->value.data[0];

        double result = pow(vbase, exponent);

        if (isnan(result))
                return graph_error(MLINC_NAN_ERROR);

        GraphNode* out = node(result);

        out->left = base;
        out->extra = tensor_scalar(exponent);
        out->backward = backward_pow;

        out->op = OP_POW;

        retain(base);

        return out;
}

void backward_log(GraphNode* self) {
        double g = self->grad.data[0];

        self->left->grad.data[0] += (1.0 / self->left->value.data[0]) * g;
}

GraphNode* log_node(GraphNode* input) {
        if (!input)
                return graph_error(MLINC_NULL_POINTER_ERROR);

        double vinput = input->value.data[0];

        if (vinput <= 0.0)
                return graph_error(MLINC_INVALID_OPERATION_ERROR);

        GraphNode* out = node(log(vinput));

        out->left = input;
        out->backward = backward_log;

        out->op = OP_LOG;

        retain(input);

        return out;
}

void backward_exp(GraphNode* self) {
        double g = self->grad.data[0];

        self->left->grad.data[0] += self->value.data[0] * g;
}

GraphNode* exp_node(GraphNode* input) {
        double vinput = input->value.data[0];

        double result = exp(vinput);

        if (isinf(result))
                return graph_error(MLINC_OVERFLOW_ERROR);
        
        GraphNode* out = node(result);

        out->left = input;
        out->backward = backward_exp;

        out->op = OP_EXP;

        retain(input);

        return out;
}