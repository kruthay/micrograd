# Acknowledgments:
micrograd is inspired from https://github.com/karpathy/micrograd 

It's written in Swift. 

# Description:


This is a command-line application to run on macOS. Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as main.swift shows. Potentially useful for educational purposes.


nn.swift consists of 3 Modules. Neuron, Layer and MLP 

engine.swift consists of a Value Type, which is a Scalar type.



# Training a neural net:

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
