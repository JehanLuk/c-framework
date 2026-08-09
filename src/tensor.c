#include <stdlib.h>
#include <string.h>

#include "tensor.h"
#include "errorcheck.h"

// Auxiliar function for errors
Tensor tensor_error(MLInCERROR err) {
    mlinc_errno = err;

    Tensor empty = {0};
    return empty;
}

static int tensor_validate_coords(const Tensor* t, const int* coords) {
    if (!t) {
        mlinc_errno = MLINC_NULL_POINTER_ERROR;
        return 0;
    }

    if (!coords) {
        mlinc_errno = MLINC_NULL_POINTER_ERROR;
        return 0;
    }

    for (int i = 0; i < t->ndim; i++) {
        if (coords[i] < 0 ||
            coords[i] >= t->shape[i]) {

            mlinc_errno = MLINC_INDEX_OUT_OF_BOUNDS_ERROR;

            return 0;
        }
    }

    return 1;
}

// Tensor functions

Tensor tensor_create(int ndim, const int* shape) {

    if (ndim < 0) {
        return tensor_error(MLINC_INVALID_DIMENSION_ERROR);
    }

    if (ndim > 0 && !shape) {
        return tensor_error(MLINC_INVALID_DIMENSION_ERROR);
    }

    Tensor t;
    t.ndim = ndim;
    t.size = 1;

    if (ndim > 0) {
        t.shape = malloc(ndim * sizeof(int));

        if (!t.shape)
            return tensor_error(MLINC_OUT_OF_MEMORY_ERROR);

        for(int i = 0; i < ndim; i++) {
            t.shape[i] = shape[i];
            t.size *= shape[i];
        }
    }
    else {
        t.shape = NULL;
    }

    t.data = calloc(t.size, sizeof(double));

    if (!t.data) {
        free(t.shape);

        return tensor_error(MLINC_OUT_OF_MEMORY_ERROR);
    }

    return t;
}

Tensor tensor_scalar(double value) {
    Tensor t;
    
    t.ndim = 0;
    t.shape = NULL;
    t.size = 1;

    t.data = malloc(sizeof(double));
    t.data[0] = value;

    return t;
}

void tensor_free(Tensor* t) {
    if (!t)
        return;

    free(t->shape);
    free(t->data);

    t->shape = NULL;
    t->data = NULL;

    t->ndim = 0;
    t->size = 0;
}

int tensor_index(const Tensor* t, const int* coords) {
    if (!t) {
        mlinc_errno = MLINC_NULL_POINTER_ERROR;
        return -1;
    }

    if (!coords) {
        mlinc_errno = MLINC_NULL_POINTER_ERROR;
        return -1;
    }

    int index = 0;
    int stride = 1;

    for (int i = t->ndim - 1; i >= 0; i--) {

        if (coords[i] < 0 ||
            coords[i] >= t->shape[i]) {

            mlinc_errno = MLINC_INDEX_OUT_OF_BOUNDS_ERROR;

            return -1;
        }

        index += coords[i] * stride;
        stride *= t->shape[i];
    }

    return index;
}

double tensor_get(const Tensor* t, const int* coords) {
    if (!tensor_validate_coords(t, coords))
        return 0.0;

    return t->data[tensor_index(t, coords)];
}

void tensor_set(Tensor *t, const int* coords, double value) {
    if (!tensor_validate_coords(t, coords))
        return;

    t->data[tensor_index(t, coords)] = value;
}