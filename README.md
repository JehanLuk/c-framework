# MLInC — Machine Learning in C

![Build](https://img.shields.io/badge/build-CMake-green)
![Status](https://img.shields.io/badge/status-experimental-B06500)
![Autograd](https://img.shields.io/badge/feature-autograd-purple)
![Level](https://img.shields.io/badge/level-low--level%20ML-red)

An educational implementation of a Machine Learning and Automatic Differentiation (**Autograd**) framework built entirely from scratch in C.

The goal of this project is to gain a deep understanding of how modern frameworks such as PyTorch and TensorFlow work internally by manually implementing their core components: tensors, computational graphs, backpropagation, and optimization algorithms.

> **Note:** Practical applications will be made in future versions.

---

# About the Project

MLInC is an experimental framework that currently implements:

* Automatic Differentiation (*Reverse-Mode Autodiff*)
* Computational Graphs
* Backpropagation
* Gradient Descent
* Multidimensional Tensors
* Basic Mathematical Operations
* Matrix Operations
* Error Handling
* Manual Memory Management

# Tensor System

The framework provides a generic tensor structure:

```c
typedef struct {
    int ndim;
    int* shape;
    int size;
    double* data;
} Tensor;
```

Where:

* `ndim` represents the number of dimensions
* `shape` represents the tensor shape
* `size` represents the total number of elements
* `data` stores the values in contiguous memory

Examples:

```text
Scalar
ndim = 0

Vector
shape = {4}
ndim = 1

Matrix
shape = {3,4}
ndim = 2
```

## Tensor Operations

### Creation and Memory Management

* `tensor_create()`
* `tensor_scalar()`
* `tensor_free()`

### Access

* `tensor_index()`
* `tensor_get()`
* `tensor_set()`

### Element-wise Operations

* `tensor_add()`
* `tensor_sub()`
* `tensor_mul()`
* `tensor_div()`

### Matrix Operations

* `tensor_transpose()`
* `tensor_matmul()`

---

# Autograd System

Each value in a computation is represented by a graph node:

```c
typedef struct GraphNode {
    Tensor value;
    Tensor grad;
    Tensor extra;

    struct GraphNode* left;
    struct GraphNode* right;

    void (*backward)(struct GraphNode*);

    Operation op;

    int ref_count;
} GraphNode;
```

Each node stores:

* The computed value
* Its accumulated gradient
* The operation that generated it
* References to dependency nodes
* A local backpropagation function

---

## Supported Operations

### Scalar Operations

* `add_node()`
* `sub_node()`
* `mul_node()`
* `div_node()`
* `pow_node()`
* `log_node()`
* `exp_node()`

Each operation provides its own backward implementation:

```text
Forward Pass
      ↓
Computational Graph
      ↓
Backward Pass
      ↓
Gradients
```

---

# Backpropagation

The framework uses:

### Topological Sorting

```c
topo(...)
```

to generate a valid propagation order.

### Backward Pass

```c
backward(loss)
```

which:

1. Builds a topological ordering of the graph
2. **Sets:**

```text
∂loss/∂loss = 1
```

3. Traverses the graph in reverse order
4. Accumulates gradients automatically

---

# Optimization

Currently the framework implements:

### Gradient Descent

```c
step(params, count, lr)
```

Updating parameters using:

```text
weight = weight - learning_rate × gradient
```

---

# Memory Management

The project implements reference counting:

```c
retain(node);
release(node);
```

allowing node reuse while preventing memory leaks.

---

# Error Handling

The framework includes a custom error system:

```c
MLInCERROR
```

including:

* `MLINC_NULL_POINTER_ERROR`
* `MLINC_OUT_OF_MEMORY_ERROR`
* `MLINC_INVALID_DIMENSION_ERROR`
* `MLINC_SHAPE_MISMATCH_ERROR`
* `MLINC_DIVISION_BY_ZERO_ERROR`
* `MLINC_INVALID_OPERATION_ERROR`
* `MLINC_OVERFLOW_ERROR`
* `MLINC_NAN_ERROR`

Global error variable:

```c
extern MLInCERROR mlinc_errno;
```

---

# Graph Export

The project integrates with GraphViz:

```c
graph_export(...)
```

allowing visualization of computational graphs generated during training.

Generated `.dot` files are exported to:

```text
epochs/
```

and can be rendered using GraphViz.

---

# Example

Simple Linear Regression:

```c
GraphNode* w = node(-3.0);
GraphNode* b = node(10.0);

GraphNode* x = node(2.0);
GraphNode* target = node(12.0);

for (int epoch = 0; epoch < 500; epoch++) {

    GraphNode* wx = mul_node(w, x);
    GraphNode* pred = add_node(wx, b);

    GraphNode* loss = mse(pred, target);

    backward(loss);

    GraphNode* params[] = {w, b};

    step(params, 2, 0.01);

    release(loss);
}
```

---

# Building the Project

MLInC uses [CMake](https://cmake.org/) as its build system.

### Requirements
- GCC (MinGW/MSYS2 on Windows)
- CMake 3.16+

### Clone the Repository

`git clone https://github.com/<your-username>/MLInC.git`


### Configure the Build

`cmake -B build`

### Compile
`cmake --build build`

This generates the executable inside the build directory.

---
### To Run:

**On Windows:**

`./build/mlinc.exe`

**On Linux/macOS:**

`./build/mlinc`

### To completely rebuild the project:

`cmake --build build --clean-first`

> Use to clean the first build

***Or remove the build directory and configure again:***

```rm -rf build
cmake -B build
cmake --build build```

---

# Development Roadmap

## Short-Term Goals

* Broadcasting
* Reduction Operations (`sum`, `mean`)
* ReLU
* Sigmoid
* Tanh
* Softmax
* Batch Operations

## Mid-Term Goals

* Dense Layers (Linear)
* MLP (Multi-Layer Perceptron)
* Dataset API
* DataLoader

## Long-Term Goals

* Convolutions
* CNN Support
* GPU Backend
* Model Serialization
* Batch Training

---

# Inspirations

* [Micrograd — Andrej Karpathy](https://github.com/karpathy/micrograd)
* [PyTorch](https://github.com/pytorch/pytorch)
* [TensorFlow](https://github.com/tensorflow/tensorflow)
* [TinyGrad](https://github.com/tinygrad/tinygrad)
* [NumPy](https://github.com/numpy/numpy)
