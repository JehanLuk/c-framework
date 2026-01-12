# C-Autograd — Autograd e Machine Learning do Zero em C

Uma implementação **didática e minimalista** de um sistema de diferenciação automática (*reverse-mode autodiff*, ou **autograd**) em C, com suporte a treinamento simples (ML) e base para redes neurais, visando a criação de um framework.

Este projeto é inspirado por implementações educacionais como o **micrograd** de Andrej Karpathy, que constrói um engine de autograd inteiro em poucas linhas de código, permitindo treinar modelos simples com gradiente descendente.

---

## 🚀 O que é este projeto

Este repositório contém:

✔️ Um **motor de autograd** em C — constrói um grafo computacional  
✔️ Operações matemáticas básicas com derivadas (`add`, `sub`, `mul`, `pow`, `log`)  
✔️ Backpropagation via topological sort  
✔️ Loop de treinamento com gradiente descendente  
✔️ Um exemplo de **regressão linear treinável**  
✔️ Base para estender para redes neurais

📌 O objetivo não é competição de performance, e sim **entendimento profundo** da lógica interna de ML.

---

## 🧠 O que você pode aprender com este projeto

Ele serve como uma **sala de aula prática viva** para:

- Diferenciação automática (*reverse-mode autodiff*)
- Grafos computacionais e backpropagation
- Gradiente descendente e otimização
- Implementação de ML do zero sem dependências externas
- Fundamentos de estruturas de dados em C (ponteiros, structs, callbacks)

Esse tipo de implementação é similar à base de grandes frameworks como PyTorch — que também constroem um grafo e propagam gradientes automaticamente — embora em C++ e com otimizações profundas. :contentReference[oaicite:1]{index=1}

---

## 🧱 Como funciona por baixo dos panos

### 🟢 Nó (`Node`)

Cada operação ou valor é armazenado como um `Node`:

- `value`: valor numérico do nó
- `grad`: gradiente acumulado
- `left`, `right`: nós dependentes (grafo)
- `backward`: função que sabe como propagar gradiente local

---

### 🔁 Construção do grafo e backpropagation

1. O forward constrói um grafo de dependências automaticamente  
2. A função `topo()` ordena os nós em uma sequência válida  
3. `backward(loss)` caminha a lista do final para o início  
4. Cada nó aplica sua derivada local multiplicada pelo gradiente acumulado

---

### 📊 Treino com gradiente descendente

O loop de treinamento faz:

forward → backward → gradient descent step

yaml
Copiar código

Com a loss definida como **MSE (Mean Squared Error)**.

---

## 📦 Exemplo de uso

No `main()`, o código treina um modelo simples:

```c
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
```

Esse exemplo aprende o melhor valor para w que aproxima y ≈ w * x.

📚 Operações suportadas
✔ Adição (add)
✔ Subtração (sub)
✔ Multiplicação (mul)
✔ Potência (pow_node)
✔ Logaritmo (log_node)

Cada uma com seu backward apropriado.

📌 Como compilar
Compile com:

bash
Copiar código
gcc -o c_autograd main.c -lm
O -lm é necessário para a biblioteca matemática (pow, log).

🧭 O que vem em seguida
Este projeto já implementa um autograd funcional e uma forma simples de treinar parâmetros. A próxima evolução natural inclui:

[ ] Adicionar bias e múltiplos parâmetros
[ ] Suportar camadas e ativações (ReLU, Sigmoid, etc.)
[ ] Construir uma rede neural multicamada (MLP)
[ ] Criar unit tests e liberar memória corretamente
[ ] Organizar em múltiplos arquivos (.h / .c)
