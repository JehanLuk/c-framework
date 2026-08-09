#ifndef ERRORCHECK_H
#define ERRORCHECK_H

typedef enum {
    //Base check
    ML_OK,

    //Memory check
    MLINC_OUT_OF_MEMORY_ERROR,
    MLINC_NULL_POINTER_ERROR,

    //Tensor check
    MLINC_INVALID_SHAPE_ERROR,
    MLINC_INVALID_DIMENSION_ERROR,
    MLINC_SHAPE_MISMATCH_ERROR,
    MLINC_INDEX_OUT_OF_BOUNDS_ERROR,

    //Operation check
    MLINC_DIVISION_BY_ZERO_ERROR,
    MLINC_INVALID_OPERATION_ERROR,
    MLINC_NOT_IMPLEMENTED_ERROR,

    //Math check
    MLINC_OVERFLOW_ERROR,
    MLINC_UNDERFLOW_ERROR,
    MLINC_NAN_ERROR,

    //Framework check
    MLINC_GRAPH_ERROR,
    MLINC_INTERNAL_ERROR
} MLInCERROR;

extern MLInCERROR mlinc_errno;

const char* mlinc_strerror(MLInCERROR error);
void mlinc_clear_error(void);

#endif