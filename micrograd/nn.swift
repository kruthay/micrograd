//
//  nn.swift
//  micrograd
//
//  Created by Kruthay Kumar Reddy Donapati on 1/23/24.
//

import Foundation


class Neuron : CustomStringConvertible {
    var w: [Value]
    var b: Value
    
    init(_ nin: Int) {
        self.w = (0..<nin).map{ _ in Value(Double.random(in: -1...1)) }
        self.b = Value(Double.random(in: -1...1))
    }
    
    var parameters: [Value] {
        return w + [b]
    }
    
    var description: String {
        var description = "( "
        for n in 0..<w.count {
            description += "(\(n+1))" + w[n].description
        }
        description += "(b)" + b.description
        return description + " )"
    }
    
    func callAsFunction(_ x: [Double]) -> Value {
        var activation: Value = zip(w,x).map { $0 * $1 }.reduce(self.b, +)
        activation = activation.tanh()
        return activation
    }
    
    func callAsFunction(_ x: [Value]) -> Value {
        var activation: Value = zip(w,x).map { $0 * $1 }.reduce(self.b, +)
        activation = activation.tanh()
        return activation
    }
}


class Layer : CustomStringConvertible {
    
    var neurons: [Neuron]
    
    var parameters: [Value] {
        neurons.map { $0.parameters }.reduce([], +)
    }
    
    var description: String {
        var description = "\n"
        for neuron in neurons {
            description += neuron.description + "\n"
        }
        return description
    }
    
    
    init(_ nin: Int, _ nout: Int) {
        self.neurons = (0..<nout).map{ _ in Neuron(nin) }
    }
    
    func callAsFunction(_ x: [Double]) -> [Value] {
        neurons.map { $0(x) }
    }
    
    func callAsFunction(_ x: [Value]) -> [Value] {
        neurons.map { $0(x) }
    }
    
    
}

class MLP : CustomStringConvertible {
    
    var layers: [Layer]
    
    var parameters: [Value] {
        return layers.map { $0.parameters }.reduce([], +)
    }
    
    var description: String {
        var description = ""
        for layer in self.layers {
            description += layer.description + "\n\n\n"
        }
        return description
    }
    
    init(_ nin: Int, _ nouts: [Int] ) {
        let totalnRequired = [nin] + nouts
        self.layers = (0..<nouts.count).map { Layer(totalnRequired[$0], totalnRequired[$0 + 1]) }
    }
    
    func callAsFunction(_ x: [Double]) -> [Value] {
        var out = layers.first!(x)
        for layer in layers[1...] {
            out = layer(out)
        }
        return out
    }
    
}
