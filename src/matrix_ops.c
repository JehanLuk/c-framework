#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "tensor.h"
#include "errorcheck.h"

// Auxiliar function for errors
Tensor tensor_error(MLInCERROR err) {
    mlinc_errno = err;

    Tensor empty = {0};
    return empty;
}

//TODO: Matrix Operations

Tensor tensor_add (const Tensor* a, const Tensor* b) {
    if (!a || !b)
        return tensor_error(MLINC_NULL_POINTER_ERROR);

    if (a->ndim != b->ndim)
        return tensor_error(MLINC_SHAPE_MISMATCH_ERROR);
    
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i])
            return tensor_error(MLINC_SHAPE_MISMATCH_ERROR);
    }

    Tensor out = tensor_create(a->ndim, a->shape);

    if (!out.data)
        return out;

    for (int i = 0; i < a->size; i++) {
        out.data[i] = a->data[i] + b->data[i];
    }

    return out;
};

Tensor tensor_sub (const Tensor* a, const Tensor* b) {
    if (!a || !b)
        return tensor_error(MLINC_NULL_POINTER_ERROR);

    if (a->ndim != b->ndim)
        return tensor_error(MLINC_SHAPE_MISMATCH_ERROR);
    
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i])
            return tensor_error(MLINC_SHAPE_MISMATCH_ERROR);
    }

    Tensor out = tensor_create(a->ndim, a->shape);

    if (!out.data)
        return out;

    for (int i = 0; i < a->size; i++) {
        out.data[i] = a->data[i] - b->data[i];
    }

    return out;
};

Tensor tensor_mul (const Tensor* a, const Tensor* b) {
    if (!a || !b)
        return tensor_error(MLINC_NULL_POINTER_ERROR);

    if (a->ndim != b->ndim)
        return tensor_error(MLINC_SHAPE_MISMATCH_ERROR);
    
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i])
            return tensor_error(MLINC_SHAPE_MISMATCH_ERROR);
    }

    Tensor out = tensor_create(a->ndim, a->shape);

    if (!out.data)
        return out;

    for (int i = 0; i < a->size; i++) {
        out.data[i] = a->data[i] * b->data[i];
    }

    return out;
};

Tensor tensor_div (const Tensor* a, const Tensor* b) {
    if (!a || !b)
        return tensor_error(MLINC_NULL_POINTER_ERROR);

    if (a->ndim != b->ndim)
        return tensor_error(MLINC_SHAPE_MISMATCH_ERROR);
    
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i])
            return tensor_error(MLINC_SHAPE_MISMATCH_ERROR);
    }

    Tensor out = tensor_create(a->ndim, a->shape);

    if (!out.data)
        return out;

    for (int i = 0; i < a->size; i++) {
        if (b->data[i] == 0.0) {
            tensor_free(&out);
            return tensor_error(MLINC_DIVISION_BY_ZERO_ERROR);
        }

        out.data[i] = a->data[i] / b->data[i];
    }

    return out;
};

Tensor tensor_matmul (Tensor* a, Tensor* b) {
    if (!a || !b)
        return tensor_error(MLINC_NULL_POINTER_ERROR);

    if (a->ndim != 2 || b->ndim != 2)
        return tensor_error(MLINC_INVALID_DIMENSION_ERROR);

    int a_rows = a->shape[0];
    int a_colns = a->shape[1];

    int b_rows = b->shape[0];
    int b_colns = b->shape[1];

    if (a_colns != b_rows)
        return tensor_error(MLINC_SHAPE_MISMATCH_ERROR);

    int out_shapes[2] = {a_rows, b_colns};

    Tensor out = tensor_create(2, out_shapes);

    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_colns; j++) {
            double sum = 0.0;

            for (int k = 0; k < a_colns; k++){
                double va = tensor_get(a, (int[]){i, k});
                double vb = tensor_get(b, (int[]){k, j});

                sum += va * vb;
            }

            tensor_set (&out, (int[]){i, j}, sum);
        }
    }

    return out;
}

Tensor tensor_transpose (const Tensor* t) {
    if (!t)
        return tensor_error(MLINC_NULL_POINTER_ERROR);

    if (t->ndim != 2)
        return tensor_error(MLINC_INVALID_DIMENSION_ERROR);

    int out_shape[2] = {t->shape[1], t->shape[0]};

    Tensor out = tensor_create(2, out_shape);

    for (int i = 0; i < t->shape[0]; i++) {
        for (int j = 0; j < t->shape[1]; j++) {
            double value = tensor_get(t, (int[]){i, j});

            tensor_set(&out, (int[]){j, i}, value);
        }
    }

    return out;
}