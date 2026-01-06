#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//Node struct definition and creation.
typedef struct Node{
        double value;
        double grad;

        double extra;

        struct Node* right;
        struct Node* left;

        void (*backward)(struct Node*);
} Node;

Node* node(double value) {
        Node* n = (Node*)malloc(sizeof(Node));

        n->value = value;
        n->grad = 0.0;
        n->right = NULL;
        n->left = NULL;
        n->backward = NULL;

        return n;
}

//Operations

void backward_add(Node* self) {
        self->left->grad += self->grad;
}

Node* add(Node* a) {
        Node* cons = node(1.0);
        Node* out = node(a->value + cons->value);

        out->left = a;
        out->right = cons;
        out->backward = backward_add;

        return out;
}

void backward_sub(Node* self) {
        self->left->grad += 1.0 * self->grad;
        self->right->grad += -1.0 * self->grad;
}

Node* sub(Node* a, Node* b) {
        Node* out = node(a->value - b->value);

        out->left = a;
        out->right = b;
        out->backward = backward_sub;

        return out;
}

void backward_mul(Node* self) {
        self->left->grad += self->right->value * self->grad;
        self->right->grad += self->left->value * self->grad;
}

Node* mul(Node* x, Node* y) {
        Node* out = node(x->value * y->value);

        out->left = x;
        out->right = y;
        out->backward = backward_mul;

        return out;
}

void backward_pow(Node* self) {
        double x = self->left->value;
        double k = self->extra;

        self->left->grad += k * pow(x, k - 1.0) * self->grad;
}

Node* pow_node(Node* x, double k) {
        Node* out = node(pow(x->value, k));

        out->left = x;
        out->extra = k;
        out->backward = backward_pow;

        return out;
}

void backward_log(Node* self) {
        self->left->grad += (1.0 / self->left->value) * self->grad;
}

Node* log_node(Node* b) {
        Node* out = node(log(b->value));

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

        loss->grad = 1.0;

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
                params[i]->value -= lr * params[i]->grad;
                params[i]->grad = 0.0;
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

                printf("epoch %d | loss %.4f | weight %.4f\n", epoch, loss->value, w->value);
        }
}