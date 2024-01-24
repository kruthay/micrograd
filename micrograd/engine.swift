//
//  main.swift
//  micrograd
//
//  Created by Kruthay Kumar Reddy Donapati on 1/23/24.
//
import Foundation

infix operator ** :  MultiplicationPrecedence

final class Value {
    
    var data : Double
    var children: [Value] = []
    var operations: String = ""
    var grad: Double = 0.0
    var label: String = ""
    var _backward: () -> Void = {}
    
    init(_ data: Double, children: [Value] = [], operations: String = "") {
        self.data = data
        self.children = children
        self.operations = operations
    }
}

extension Value: CustomStringConvertible {
    var description: String {
        return "\(label) d: \(String(format: "%.4f",data)), g: \(String(format: "%.4f",grad)) "
    }
}

extension Value: Hashable {
    
    static func == (lhs: Value, rhs: Value) -> Bool {
        return lhs.data == rhs.data && lhs.grad == rhs.grad && lhs.children == rhs.children
    }
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(self.data)
        hasher.combine(self.grad)
    }
    
}

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
}



