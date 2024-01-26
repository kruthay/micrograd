# Micrograd-Swift

## Acknowledgments
This project draws inspiration from [karpathy/micrograd](https://github.com/karpathy/micrograd).

## Overview
Micrograd-Swift is a lightweight macOS command-line application that focuses on implementing backpropagation (reverse-mode autodiff) through a dynamically built Directed Acyclic Graph (DAG). The codebase consists of two primary modules, each succinctly written with around 100 lines for the DAG and 50 lines for a neural networks library, providing a PyTorch-like API. The DAG operates exclusively on scalar values, deconstructing each neuron into fundamental additions and multiplications. Despite its simplicity, the application is capable of constructing entire deep neural networks for binary classification, serving as a valuable educational resource.

## Modules

### nn.swift
This module encompasses three key components:
- **Neuron**: Represents an individual neuron.
- **Layer**: Defines a layer of neurons.
- **MLP (Multi-Layer Perceptron)**: Represents a neural network with a PyTorch-like API.

### engine.swift
This module introduces the `Value` type, which serves as the backbone for representing scalar values in the DAG.

### Tensor.swift
The `Tensor` class functions as a vector and can be used for various operations. Here's a brief overview:

```swift
class Tensor {
    
    // Properties and Initializers
    
    // ... (Details as per Tensor.swift file)
    
    // View Function
    func view(to shape: Int...) -> Bool {
        // ... (Details as per Tensor.swift file)
    }
    
    // Subscripts
    
    // ... (Details as per Tensor.swift file)
    
    // Backward Pass
    func backward() {
        // ... (Details as per Tensor.swift file)
    }
    
    // ... (Other methods and functionalities as per Tensor.swift file)
}
```

### Training a neural net:

main.swift consists of a sample small neural network. 

```swift
import Foundation
var mlp = MLP(3, [4, 4, 1])

var xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
var ys = [1.0, -1.0, -1.0, 1.0]
var ypred : [Value] = []

var loss : Value = Value(0)
var lr : Double

for _ in 0...400 {
    ypred = xs.map { value in
        return mlp(value).first!
    }
    loss = zip(ys,ypred).map { ( $0 - $1 ) ** 2 }.reduce(Value(0), +)
    mlp.parameters.forEach { $0.grad = 0 }
    loss.backward()
    lr = 0.1
    mlp.parameters.forEach { $0.data -= lr * $0.grad }
}

```

# License
MIT
