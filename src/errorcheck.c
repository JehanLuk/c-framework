#include "errorcheck.h"

MLInCERROR mlinc_errno = ML_OK;

void mlinc_clear_error(void) {
    mlinc_errno = ML_OK;
}

const char* mlinc_strerror(MLInCERROR error) {
    switch (error) {
        case ML_OK:
            return "No error(s)";
        case MLINC_OUT_OF_MEMORY_ERROR:
            return "Out of memory";
        case MLINC_NULL_POINTER_ERROR:
            return "Null pointer(s)";
        case MLINC_INVALID_SHAPE_ERROR:
            return "Invalid shape(s)";
        case MLINC_INVALID_DIMENSION_ERROR:
            return "Invalid dimension(s)";
        case MLINC_SHAPE_MISMATCH_ERROR:
            return "Shape(s) mismatch";
        case MLINC_INDEX_OUT_OF_BOUNDS_ERROR:
            return "Index out of bounds";
        case MLINC_DIVISION_BY_ZERO_ERROR:
            return "Division by zero";
        case MLINC_INVALID_OPERATION_ERROR:
            return "Invalid operation(s)";
        case MLINC_NOT_IMPLEMENTED_ERROR:
            return "Not implemented";
        case MLINC_OVERFLOW_ERROR:
            return "Overflow";
        case MLINC_UNDERFLOW_ERROR:
            return "Underflow";
        case MLINC_NAN_ERROR:
            return "NaN generated";
        case MLINC_GRAPH_ERROR:
            return "Graph error";
        case MLINC_INTERNAL_ERROR:
            return "Internal error";
        default:
            return "Unknown error";
    }
}