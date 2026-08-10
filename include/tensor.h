#ifndef TENSOR_H
#define TENSOR_H

#include "errorcheck.h"

//Tensor struct for dataset storing and math base
typedef struct {
    int ndim;
    int* shape;
    int size;
    double* data;
} Tensor;

Tensor tensor_error(MLInCERROR err);

Tensor tensor_create(int ndim, const int* shape);
Tensor tensor_scalar(double value);

void tensor_free(Tensor* t);

int tensor_index(const Tensor* t, const int* coords);

double tensor_get(const Tensor* t, const int* coords);
void tensor_set(Tensor* t, const int* coords, double value);

#endif