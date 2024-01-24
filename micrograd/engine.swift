//
//  main.swift
//  micrograd
//
//  Created by Kruthay Kumar Reddy Donapati on 1/23/24.
//
import Foundation

// Exponent operator
infix operator ** :  MultiplicationPrecedence

// Scalar representation of data which includes gradients and gradient function.
final class Value {
    // data value
    var data : Double
    
    // children property to store the children Values, required for backprop
    var children: [Value] = []
    
    // operations property is made to be used for GraphViz, which is currently buggy for Swift
    var operations: String = ""
    
    // grad property to store the gradient value
    var grad: Double = 0.0
    
    // label for the GraphViz
    var label: String = ""
    
    // _backward stores the gradient function
    var _backward: () -> Void = {}
    
    init(_ data: Double, children: [Value] = [], operations: String = "") {
        self.data = data
        self.children = children
        self.operations = operations
    }
}

// String Convertible to provide accurate description for Value type
extension Value: CustomStringConvertible {
    var description: String {
        return "\(label) d: \(String(format: "%.4f",data)), g: \(String(format: "%.4f",grad)) "
    }
}

//Required for the topological sort function
extension Value: Hashable {
    
    static func == (lhs: Value, rhs: Value) -> Bool {
        return lhs.data == rhs.data && lhs.grad == rhs.grad && lhs.children == rhs.children
    }
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(self.data)
        hasher.combine(self.grad)
    }
    
}

// Used to get += methods by default
extension Value: AdditiveArithmetic {
    
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
    
}

// Used to get protocol based support for Numeric Variables and Error Values
extension Value : Numeric {
    var magnitude: Double {
        return Swift.abs(data)
    }
    
    convenience init(integerLiteral value: Int) {
        self.init(Double(value))
    }
    
    typealias Magnitude = Double
    
    convenience init?<T>(exactly source: T) where T : BinaryInteger {
        self.init(Double(source))
    }
    
    typealias IntegerLiteralType = Int
    
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
    
    static func *= (lhs: inout Value, rhs: Value) {
        lhs = lhs * rhs
    }
    
    static func *= (lhs: inout Value, rhs: Double) {
        lhs = lhs * rhs
    }
    
    
    static func ** (lhs: Value, rhs: Double) -> Value {
        let out = Value(pow(lhs.data, rhs), children: [lhs], operations: "**")
        out._backward = {
            lhs.grad += rhs * pow(lhs.data, rhs - 1 ) * out.grad
        }
        return out
    }
}

// Activation Functions
extension Value {
    // tanh activation function and it's gradient function
    func tanh() -> Value {
        let val = exp(2.0 * self.data)
        let out = Value((val - 1 ) / (val + 1), children: [self], operations: "tanh")
        out._backward = {
            self.grad += ( 1 - pow(out.data, 2.0) ) * out.grad
        }
        return out
        
    }
}

// Backward pass
extension Value {
    // Topological sort is used to create a DAG and to apply backward pass on the reverse sorted order
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
}



