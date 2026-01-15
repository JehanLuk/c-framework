#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//Tensor struct for dataset storing and math base.
typedef struct {
        int size;
        double* data;
} Tensor;

//Node struct definition and creation.
typedef struct Node{
        Tensor value;
        Tensor grad;

        Tensor extra;

        struct Node* right;
        struct Node* left;

        void (*backward)(struct Node*);
} Node;

Tensor tensor_scalar(double v) {
        Tensor t;
        t.size = 1;
        t.data = malloc(sizeof(double));
        t.data[0] = v;
        return t;
}

Node* node(double value) {
        Node* n = malloc(sizeof(Node));

        n->value = tensor_scalar(value);
        n->grad = tensor_scalar(0.0);
        n->right = NULL;
        n->left = NULL;
        n->backward = NULL;

        return n;
}

//Operations

void backward_add(Node* self) {
        double g = self->grad.data[0];

        self->left->grad.data[0] += g;
}

Node* add(Node* a) {
        double va = a->value.data[0];
        Node* cons = node(1.0);
        double vcons = cons->value.data[0];
        Node* out = node(va + vcons);

        out->left = a;
        out->right = cons;
        out->backward = backward_add;

        return out;
}

void backward_sub(Node* self) {
        double g = self->grad.data[0];

        self->left->grad.data[0] += 1.0 * g;
        self->right->grad.data[0] += -1.0 * g;
}

Node* sub(Node* a, Node* b) {
        double va = a->value.data[0];
        double vb = b->value.data[0];
        
        Node* out = node(va - vb);

        out->left = a;
        out->right = b;
        out->backward = backward_sub;

        return out;
}

void backward_mul(Node* self) {
        double g = self->grad.data[0];
        
        self->left->grad.data[0] += self->right->value.data[0] * g;
        self->right->grad.data[0] += self->left->value.data[0] * g;
}

Node* mul(Node* x, Node* y) {
        double vx = x->value.data[0];
        double vy = y->value.data[0];

        Node* out = node(vx * vy);

        out->left = x;
        out->right = y;
        out->backward = backward_mul;

        return out;
}

void backward_pow(Node* self) {
        double g = self->grad.data[0];

        double x = self->left->value.data[0];
        double k = self->extra.data[0];

        self->left->grad.data[0] += k * pow(x, k - 1.0) * g;
}

Node* pow_node(Node* x, double k) {
        double vx = x->value.data[0];

        Node* out = node(pow(vx, k));

        out->left = x;
        out->extra = tensor_scalar(k);
        out->backward = backward_pow;

        return out;
}

void backward_log(Node* self) {
        double g = self->grad.data[0];

        self->left->grad.data[0] += (1.0 / self->left->value.data[0]) * g;
}

Node* log_node(Node* b) {
        double vb = b->value.data[0];

        Node* out = node(log(vb));

        out->left = b;
        out->backward = backward_log;

        return out;
}

//TOPO (Topological sorting) and backward/backpropagation

void topo(Node* n, Node** list, int* size) {
        if (!n) return;

        topo(n->left, list, size);
        topo(n->right, list, size);

        list[(*size)++] = n;
}

void backward(Node* loss) {
        Node* order [1000];
        int size = 0;

        topo(loss, order, &size);

        loss->grad.data[0] = 1.0;

        for (int i = size - 1; i >= 0; i--) {
                if(order[i]->backward) {
                        order[i]->backward(order[i]);
                }
        }
}

//Loss (using MSE (Mean Squared Error) as function) and optimization

Node* mse(Node* pred, Node* target) {
        Node* diff = sub(pred, target);
        return pow_node(diff, 2);
}

void step(Node** params, int count, double lr) {
        for (int i = 0; i < count; i++) {
                params[i]->value.data[0] -= lr * params[i]->grad.data[0];
                params[i]->grad.data[0] = 0.0;
        }
}

int main() {
        Node* w = node(0.5);
        Node* x = node(3.0);
        Node* y = node(2.0);

        for (int epoch = 0; epoch < 100; epoch++) {
                Node* pred = mul(w, x);
                Node* loss = mse(pred, y);

                backward(loss);
                step(&w, 1, 0.01);

                printf("epoch %d | loss %.4f | weight %.4f\n", epoch, loss->value.data[0], w->value.data[0]);
        }
}