# MLInC — Machine Learning in C

![Autograd](https://img.shields.io/badge/autograd-reverse--mode-purple)
![Status](https://img.shields.io/badge/status-alpha-orange)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)

Uma implementação educacional de um framework de Machine Learning e Diferenciação Automática (**Autograd**) construída do zero em C.

O objetivo do projeto é compreender profundamente como frameworks modernos como PyTorch e TensorFlow funcionam internamente, implementando manualmente seus principais componentes: tensores, grafos computacionais, backpropagation e otimização.

> OBS: Futuramente serão feitos usos práticos

## Sobre o projeto

O MLInC é um framework experimental que implementa:

* Diferenciação automática (*reverse-mode autodiff*)
* Grafos computacionais
* Backpropagation
* Gradiente descendente
* Tensores multidimensionais
* Operações matemáticas básicas
* Operações matriciais
* Tratamento de erros
* Gerenciamento manual de memória


# Sistema de Tensores

O framework possui uma estrutura genérica de tensor:

```c
typedef struct {
    int ndim;
    int* shape;
    int size;
    double* data;
} Tensor;
```

Onde:

* `ndim` representa o número de dimensões
* `shape` representa o formato do tensor
* `size` representa a quantidade total de elementos
* `data` armazena os valores em memória contínua

Exemplos:

```text
Escalar
ndim = 0

Vetor
shape = {4}
ndim = 1

Matriz
shape = {3,4}
ndim = 2
```


## Operações de Tensor

### Criação e gerenciamento

* tensor_create()
* tensor_scalar()
* tensor_free()

### Acesso

* tensor_index()
* tensor_get()
* tensor_set()

### Operações elemento a elemento

* tensor_add()
* tensor_sub()
* tensor_mul()
* tensor_div()

### Operações matriciais

* tensor_transpose()
* tensor_matmul()

# Sistema de Autograd

Cada valor da computação é representado por um nó do grafo:

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

Cada nó armazena:

* valor
* gradiente
* operação que o gerou
* dependências no grafo
* função de backpropagation


## Operações suportadas

### Escalares

* add_node()
* sub_node()
* mul_node()
* div_node()
* pow_node()
* log_node()
* exp_node()

Cada operação possui sua implementação de backward:

```text
forward -> grafo computacional -> backward -> gradientes
```


# Backpropagation

O algoritmo utiliza:

### Ordenação topológica

```c
topo(...)
```

para gerar a sequência correta de propagação.

### Backward

```c
backward(loss)
```

que:

1. Constrói a ordem topológica
2. **Define:**

```text
∂loss/∂loss = 1
```

3. Percorre o grafo de trás para frente
4. Acumula gradientes automaticamente


# Otimização

Atualmente o framework implementa:

### Gradient Descent

```c
step(params, count, lr)
```

Atualizando:

```text
peso = peso - learning_rate × gradiente
```

# Gerenciamento de memória

O projeto implementa contagem de referências:

```c
retain(node)
release(node)
```

permitindo reutilização de nós sem vazamentos de memória.

# Sistema de erros

O framework utiliza um sistema próprio de erros:

```c
MLInCERROR
```

incluindo:

* MLINC_NULL_POINTER_ERROR
* MLINC_OUT_OF_MEMORY_ERROR
* MLINC_INVALID_DIMENSION_ERROR
* MLINC_SHAPE_MISMATCH_ERROR
* MLINC_DIVISION_BY_ZERO_ERROR
* MLINC_INVALID_OPERATION_ERROR
* MLINC_OVERFLOW_ERROR
* MLINC_NAN_ERROR

Erro global:

```c
extern MLInCERROR mlinc_errno;
```

---

# Exportação do grafo

O projeto possui integração com GraphViz:

```c
graph_export(...)
```

permitindo visualizar o grafo computacional gerado durante o treinamento.

Arquivos `.dot` são exportados para:

```text
epochs/
```

e podem ser renderizados com GraphViz.

# Exemplo

Regressão linear simples:

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

# Roadmap de Objetivos

## Curto prazo

* Broadcasting
* Reduções (sum, mean)
* ReLU
* Sigmoid
* Tanh
* Softmax
* Batch operations

## Médio prazo

* Camadas densas (Linear)
* MLP (Multi-Layer Perceptron)
* Dataset API
* DataLoader

## Longo prazo

* Convoluções
* CNNs
* GPU Backend
* Serialização de modelos
* Treinamento em batches


# Inspirações

- [Micrograd — Andrej Karpathy](https://github.com/karpathy/micrograd)
- [PyTorch](https://github.com/pytorch/pytorch)
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [TinyGrad](https://github.com/tinygrad/tinygrad)
- [NumPy](https://github.com/numpy/numpy)