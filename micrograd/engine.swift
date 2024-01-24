//
//  main.swift
//  micrograd
//
//  Created by Kruthay Kumar Reddy Donapati on 1/23/24.
//
import Foundation

infix operator ** :  MultiplicationPrecedence

class Value: CustomStringConvertible, Equatable, Hashable {
    
    var description: String {
        return "\(label) d: \(String(format: "%.4f",data)), g: \(String(format: "%.4f",grad)) "
    }
    
    var data : Double
    var children: [Value]
    var operations: String
    var grad: Double
    var label: String = ""
    
    var _backward: () -> Void
    
    init(_ data: Double, children: [Value] = [], operations: String = "") {
        self.data = data
        self.children = children
        self.operations = operations
        self.grad = 0.00
        self._backward = {}
    }
    
    static func == (lhs: Value, rhs: Value) -> Bool {
            return lhs.data == rhs.data && lhs.children == rhs.children && lhs.operations == rhs.operations && lhs.grad == rhs.grad
        }
    
    static func + (lhs: Value, rhs: Value) -> Value {
        let out = Value(lhs.data + rhs.data, children: [lhs, rhs], operations: "+")
        out._backward = {
            lhs.grad += out.grad
            rhs.grad += out.grad

        }
        return out
        
    }
    
    static func + (lhs: Value, rhs: Double) -> Value {
        return lhs + Value(rhs)
        
    }
    
    static func + (lhs: Double, rhs: Value) -> Value {
        return Value(lhs) + rhs
    }
    
    static func - (lhs: Value, rhs: Value) -> Value {
        return lhs + (-1 * rhs)
    }
    
    static func - (lhs: Value, rhs: Double) -> Value {
        return lhs - Value(rhs)
    }
    
    static func - (lhs: Double, rhs: Value) -> Value {
        return Value(lhs) - rhs
    }
    
    static func / (lhs: Value, rhs: Value) -> Value {
        return lhs * (rhs ** (-1))
    }
    
    static func / (lhs: Value, rhs: Double) -> Value {
        return lhs * (1/rhs)
    }
    
    static func / (lhs: Double, rhs: Value) -> Value {
        return lhs * (rhs ** (-1))
    }
    
    static func * (lhs: Value, rhs: Value) -> Value {
        let out = Value(lhs.data * rhs.data, children: [lhs, rhs], operations: "*")

        out._backward = {
            lhs.grad += rhs.data * out.grad
            rhs.grad += lhs.data * out.grad
        }
        
        return out
    }
    
    static func * (lhs: Value, rhs: Double) -> Value {
        return lhs * Value(rhs)
    }
    
    static func * (lhs: Double, rhs: Value) -> Value {
        return Value(lhs) * rhs
    }
    
    
    static func ** (lhs: Value, rhs: Double) -> Value {
        let out = Value(pow(lhs.data, rhs), children: [lhs], operations: "**")
        out._backward = {
            lhs.grad += rhs * pow(lhs.data, rhs - 1 ) * out.grad
        }
        return out
    }
    
    func tanh() -> Value {
        let val = exp(2.0 * self.data)
        let out = Value((val - 1 ) / (val + 1), children: [self], operations: "tanh")
        out._backward = {
            self.grad += ( 1 - pow(out.data, 2.0) ) * out.grad
        }
        return out
        
    }
    
    func backward() {
        var sorted : [Value] = []
        var visited: Set<Value> = []
        
        func buildTopologicalSort(_ vertex : Value) {
            if !visited.contains(vertex) {
                visited.insert(vertex)
                for child in vertex.children {
                    buildTopologicalSort(child)
                }
                sorted.append(vertex)
            }
        }
        buildTopologicalSort(self)
        
        self.grad = 1.0
        sorted.reverse()
        for value in sorted {
            value._backward()
        }
    }
    
    func hash(into hasher: inout Hasher) {
            hasher.combine(self.data)
            hasher.combine(self.grad)
            hasher.combine(self.operations)
        }

}



class Neuron : CustomStringConvertible {
    var w: [Value]
    var b: Value
    
    init(_ nin: Int) {
        self.w = Array(repeating: Value(Double.random(in: -1...1)), count: nin)
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
        self.neurons = Array(repeating: Neuron(nin), count: nout)
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
        for layer in n.layers {
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

var n = MLP(3, [4, 4, 1])

var xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

var ys = [1.0, -1.0, -1.0, 1.0]
var ypred : [Value] = xs.map { value in
    return n(value).first!
}

var loss : Value
var stepSize : Double
for _ in 0...1 {
    ypred = xs.map { value in
        return n(value).first!
    }
    loss = zip(ys,ypred).map { ( $0 - $1 ) ** 2 }.reduce(Value(0), +)
    n.parameters.forEach { $0.grad = 0 }
    loss.backward()
    stepSize = 0.001
    n.parameters.forEach { $0.data -= stepSize * $0.grad }
}
print(n)
