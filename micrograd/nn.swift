//
//  nn.swift
//  micrograd
//
//  Created by Kruthay Kumar Reddy Donapati on 1/23/24.
//

import Foundation

/// Represents a single neuron in a neural network layer.
class Neuron: CustomStringConvertible {
    var w: [Value] // Weights
    var b: Value    // Bias
    
    /// Initializes a neuron with random weights and bias.
    ///
    /// - Parameters:
    ///   - nin: The number of input features.
    init(_ nin: Int) {
        self.w = (0..<nin).map { _ in Value(Double.random(in: -1...1)) }
        self.b = Value(Double.random(in: -1...1))
    }
    
    /// Returns the parameters (weights and bias) of the neuron.
    var parameters: [Value] {
        return w + [b]
    }
    
    /// A string representation of the neuron.
    var description: String {
        var description = "( "
        for n in 0..<w.count {
            description += "(\(n+1))" + w[n].description
        }
        description += "(b)" + b.description
        return description + " )"
    }
    
    /// Computes the output of the neuron given an input array of Doubles.
    ///
    /// - Parameter x: An array of Doubles representing the input.
    ///
    /// - Returns: The output `Value` of the neuron.
    func callAsFunction(_ x: [Double]) -> Value {
        var activation: Value = zip(w, x).map { $0 * $1 }.reduce(self.b, +)
        activation = activation.tanh()
        return activation
    }
    
    /// Computes the output of the neuron given an input array of Values.
    ///
    /// - Parameter x: An array of Values representing the input.
    ///
    /// - Returns: The output `Value` of the neuron.
    func callAsFunction(_ x: [Value]) -> Value {
        var activation: Value = zip(w, x).map { $0 * $1 }.reduce(self.b, +)
        activation = activation.tanh()
        return activation
    }
}


/// Represents a layer in a neural network.
class Layer: CustomStringConvertible {
    
    var neurons: [Neuron]
    
    /// Returns the parameters (weights and biases) of all neurons in the layer.
    var parameters: [Value] {
        neurons.map { $0.parameters }.reduce([], +)
    }
    
    /// A string representation of the layer.
    var description: String {
        var description = "\n"
        for neuron in neurons {
            description += neuron.description + "\n"
        }
        return description
    }
    
    /// Initializes a layer with a specified number of input and output neurons.
    ///
    /// - Parameters:
    ///   - nin: The number of input features.
    ///   - nout: The number of output neurons in the layer.
    init(_ nin: Int, _ nout: Int) {
        self.neurons = (0..<nout).map { _ in Neuron(nin) }
    }
    
    /// Computes the output of the layer given an input array of Doubles.
    ///
    /// - Parameter x: An array of Doubles representing the input.
    ///
    /// - Returns: An array of `Value` instances representing the output of each neuron in the layer.
    func callAsFunction(_ x: [Double]) -> [Value] {
        neurons.map { $0(x) }
    }
    
    /// Computes the output of the layer given an input array of Values.
    ///
    /// - Parameter x: An array of Values representing the input.
    ///
    /// - Returns: An array of `Value` instances representing the output of each neuron in the layer.
    func callAsFunction(_ x: [Value]) -> [Value] {
        neurons.map { $0(x) }
    }
}


/// Represents a Multi-Layer Perceptron (MLP) neural network.
class MLP: CustomStringConvertible {
    
    var layers: [Layer]
    
    /// Returns the parameters (weights and biases) of all layers in the MLP.
    var parameters: [Value] {
        return layers.map { $0.parameters }.reduce([], +)
    }
    
    /// A string representation of the MLP.
    var description: String {
        var description = ""
        for layer in self.layers {
            description += layer.description + "\n\n\n"
        }
        return description
    }
    
    /// Initializes an MLP with a specified number of input neurons and an array of output neurons for each layer.
    ///
    /// - Parameters:
    ///   - nin: The number of input features.
    ///   - nouts: An array specifying the number of output neurons for each layer in the MLP.
    init(_ nin: Int, _ nouts: [Int]) {
        let totalnRequired = [nin] + nouts
        self.layers = (0..<nouts.count).map { Layer(totalnRequired[$0], totalnRequired[$0 + 1]) }
    }
    
    /// Computes the output of the entire neural network given an input array of Doubles.
    ///
    /// - Parameter x: An array of Doubles representing the input.
    ///
    /// - Returns: An array of `Value` instances representing the output of each neuron in the last layer.
    func callAsFunction(_ x: [Double]) -> [Value] {
        var out = layers.first!(x)
        for layer in layers[1...] {
            out = layer(out)
        }
        return out
    }
}
