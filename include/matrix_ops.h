#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include "tensor.h"

Tensor tensor_add (const Tensor* a, const Tensor* b);
Tensor tensor_sub (const Tensor* a, const Tensor* b);
Tensor tensor_mul (const Tensor* a, const Tensor* b);
Tensor tensor_div (const Tensor* a, const Tensor* b);

Tensor tensor_matmul (Tensor* a, Tensor* b);
Tensor tensor_transpose(const Tensor* t);

#endif