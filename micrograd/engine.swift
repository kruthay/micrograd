//
//  main.swift
//  micrograd
//
//  Created by Kruthay Kumar Reddy Donapati on 1/23/24.
//

import Foundation

// Exponent operator
infix operator ** :  MultiplicationPrecedence

/// Represents a scalar value that includes gradients and gradient functions.
final class Value {
    // Actual data value
    var data: Double
    
    // Children property to store the child Values, required for backpropagation
    var children: [Value] = []
    
    // String representation of operations, used for GraphViz (currently not fully functional in Swift)
    var operations: String = ""
    
    // Gradient value
    var grad: Double = 0.0
    
    // Label for GraphViz
    var label: String = ""
    
    // Closure to store the gradient function
    var _backward: () -> Void = {}
    
    /// Initializes a Value with given data, children, and operations.
    init(_ data: Double, children: [Value] = [], operations: String = "") {
        self.data = data
        self.children = children
        self.operations = operations
    }
}

// String Convertible to provide accurate description for Value type
extension Value: CustomStringConvertible {
    var description: String {
        return "\(label) d: \(String(format: "%.4f", data)), g: \(String(format: "%.4f", grad)) "
    }
}

// Required for the topological sort function
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
    
    /// Addition of two Values.
    static func + (lhs: Value, rhs: Value) -> Value {
        let out = Value(lhs.data + rhs.data, children: [lhs, rhs], operations: "+")
        out._backward = {
            lhs.grad += out.grad
            rhs.grad += out.grad
        }
        return out
    }
    
    /// Addition of a Value and a Double.
    static func + (lhs: Value, rhs: Double) -> Value {
        return lhs + Value(rhs)
    }
    
    /// Addition of a Double and a Value.
    static func + (lhs: Double, rhs: Value) -> Value {
        return Value(lhs) + rhs
    }
}

// Used to get protocol-based support for Numeric Variables and Error Values
extension Value: Numeric {
    var magnitude: Double {
        return Swift.abs(data)
    }
    
    /// Initializes a Value with an integer literal.
    convenience init(integerLiteral value: Int) {
        self.init(Double(value))
    }
    
    typealias Magnitude = Double
    
    /// Initializes a Value with a binary integer.
    convenience init?<T>(exactly source: T) where T: BinaryInteger {
        self.init(Double(source))
    }
    
    typealias IntegerLiteralType = Int
    
    /// Subtraction of two Values.
    static func - (lhs: Value, rhs: Value) -> Value {
        return lhs + (-1 * rhs)
    }
    
    /// Subtraction of a Value and a Double.
    static func - (lhs: Value, rhs: Double) -> Value {
        return lhs - Value(rhs)
    }
    
    /// Subtraction of a Double and a Value.
    static func - (lhs: Double, rhs: Value) -> Value {
        return Value(lhs) - rhs
    }
    
    /// Division of two Values.
    static func / (lhs: Value, rhs: Value) -> Value {
        return lhs * (rhs ** (-1))
    }
    
    /// Division of a Value and a Double.
    static func / (lhs: Value, rhs: Double) -> Value {
        return lhs * (1 / rhs)
    }
    
    /// Division of a Double and a Value.
    static func / (lhs: Double, rhs: Value) -> Value {
        return lhs * (rhs ** (-1))
    }
    
    /// Multiplication of two Values.
    static func * (lhs: Value, rhs: Value) -> Value {
        let out = Value(lhs.data * rhs.data, children: [lhs, rhs], operations: "*")
        
        out._backward = {
            lhs.grad += rhs.data * out.grad
            rhs.grad += lhs.data * out.grad
        }
        
        return out
    }
    
    /// Multiplication of a Value and a Double.
    static func * (lhs: Value, rhs: Double) -> Value {
        return lhs * Value(rhs)
    }
    
    /// Multiplication of a Double and a Value.
    static func * (lhs: Double, rhs: Value) -> Value {
        return Value(lhs) * rhs
    }
    
    /// Compound assignment multiplication of two Values.
    static func *= (lhs: inout Value, rhs: Value) {
        lhs = lhs * rhs
    }
    
    /// Compound assignment multiplication of a Value and a Double.
    static func *= (lhs: inout Value, rhs: Double) {
        lhs = lhs * rhs
    }
    
    /// Exponential operation on a Value with a Double exponent.
    static func ** (lhs: Value, rhs: Double) -> Value {
        let out = Value(pow(lhs.data, rhs), children: [lhs], operations: "**")
        out._backward = {
            lhs.grad += rhs * pow(lhs.data, rhs - 1) * out.grad
        }
        return out
    }
}

// Activation Functions
extension Value {
    /// Tanh activation function and its gradient function.
    func tanh() -> Value {
        let val = exp(2.0 * self.data)
        let out = Value((val - 1) / (val + 1), children: [self], operations: "tanh")
        out._backward = {
            self.grad += (1 - pow(out.data, 2.0)) * out.grad
        }
        return out
    }
}

// Backward pass
extension Value {
    /// Topological sort is used to create a Directed Acyclic Graph (DAG) for backward pass.
    func backward() {
            var sorted: [Value] = []
            var visited: Set<Value> = []
            
            func buildTopologicalSort(_ vertex: Value) {
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


