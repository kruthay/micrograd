micrograd is inspired from https://github.com/karpathy/micrograd 
It's written in Swift. 
This is a command-line application to run on macOS. 

However, entire code is present in two files called engine.swift, & nn.swift.

nn.swift consists of 3 Modules. Neuron, Layer and MLP 
engine.swift consists of a Value Type, which is a Scalar type.

Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.


# Training a neural net

main.swift consists of a sample small neural network. 

# License
MIT
