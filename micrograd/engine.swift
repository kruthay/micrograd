//
//  main.swift
//  micrograd
//
//  Created by Kruthay Kumar Reddy Donapati on 1/23/24.
//

import Foundation

// Exponent operator
infix operator ** :  MultiplicationPrecedence

/// Represents a node in a computational graph used for automatic differentiation.
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
    
    /// Initializes a `Value` with given data, children, and operations.
    ///
    /// - Parameters:
    ///   - data: The actual data value of the `Value`.
    ///   - children: An array of child `Value` instances, required for backpropagation.
    ///   - operations: A string representation of operations, used for GraphViz (currently not fully functional in Swift).
    init(_ data: Double, children: [Value] = [], operations: String = "") {
        self.data = data
        self.children = children
        self.operations = operations
    }
}


/// Conformance to the `CustomStringConvertible` protocol for `Value`.
///
/// Provides a custom string representation for instances of `Value`.
///
/// - Note: The string representation includes the `label`, `data`, and `grad` properties.
///
/// - Example:
///   ```swift
///   let value = Value(5.0, label: "example")
///   let description = value.description
///   ```
///   In this example, `description` will be a string representing the `label`, `data`, and `grad` properties of the `value`.
extension Value: CustomStringConvertible {
    
    /// The custom string representation of the `Value` instance.
    var description: String {
        return "\(label) d: \(String(format: "%.8f", data)), g: \(String(format: "%.8f", grad)) "
    }
}


/// Conformance to the `Hashable` protocol for `Value`.
///
/// Enables instances of `Value` to be used in data structures that require hashing.
///
/// - Note: The comparison for equality (`==`) is based on the equality of `data`, `grad`, and `children` properties.
///
/// - Example:
///   ```swift
///   let value1 = Value(3.0)
///   let value2 = Value(3.0)
///   let areEqual = value1 == value2
///   ```
///   In this example, `areEqual` will be `true` because the `data`, `grad`, and `children` properties of `value1` and `value2` are equal.
extension Value: Hashable {
    
    /// Compares two `Value` instances for equality.
    ///
    /// - Parameters:
    ///   - lhs: The left-hand operand `Value`.
    ///   - rhs: The right-hand operand `Value`.
    ///
    /// - Returns: `true` if the `data`, `grad`, and `children` properties of both instances are equal; otherwise, `false`.
    static func == (lhs: Value, rhs: Value) -> Bool {
        return lhs.data == rhs.data && lhs.grad == rhs.grad && lhs.children == rhs.children
    }
    
    /// Computes the hash value for a `Value` instance.
    ///
    /// - Parameters:
    ///   - hasher: The hasher to use for combining hash values.
    func hash(into hasher: inout Hasher) {
        hasher.combine(self.data)
        hasher.combine(self.grad)
    }
}



extension Value: AdditiveArithmetic {
    
    /// Performs addition of two `Value` instances.
    ///
    /// This operator calculates the result of adding the left-hand `Value` to the right-hand `Value`.
    ///
    /// - Parameters:
    ///   - lhs: The first `Value` to be added.
    ///   - rhs: The second `Value` to be added.
    ///
    /// - Returns: A new `Value` representing the result of the addition operation.
    ///
    /// - Note: The `operations` property of the resulting `Value` is set to "+" to indicate the operation performed.
    ///
    /// - Example:
    ///   ```swift
    ///   let value1 = Value(3.0)
    ///   let value2 = Value(5.0)
    ///   let result = value1 + value2
    ///   ```
    ///   In this example, `result` will be a new `Value` representing the sum of 3.0 and 5.0.
    static func + (lhs: Value, rhs: Value) -> Value {
        // Calculate the result of the addition operation.
        let out = Value(lhs.data + rhs.data, children: [lhs, rhs], operations: "+")
        
        // Define the backward closure for gradient calculation during backpropagation.
        out._backward = {
            lhs.grad += out.grad
            rhs.grad += out.grad
        }
        
        // Return the result of the addition operation.
        return out
    }
    
    /// Performs addition of a `Value` and a `Double`.
    ///
    /// This operator calculates the result of adding a `Double` to a `Value`.
    ///
    /// - Parameters:
    ///   - lhs: The `Value` to be added.
    ///   - rhs: The `Double` to be added.
    ///
    /// - Returns: A new `Value` representing the result of the addition operation.
    ///
    /// - Example:
    ///   ```swift
    ///   let value = Value(7.0)
    ///   let doubleValue = 2.0
    ///   let result = value + doubleValue
    ///   ```
    ///   In this example, `result` will be a new `Value` representing the sum of 7.0 and 2.0.
    static func + (lhs: Value, rhs: Double) -> Value {
        return lhs + Value(rhs)
    }
    
    /// Performs addition of a `Double` and a `Value`.
    ///
    /// This operator calculates the result of adding a `Value` to a `Double`.
    ///
    /// - Parameters:
    ///   - lhs: The `Double` to be added.
    ///   - rhs: The `Value` to be added.
    ///
    /// - Returns: A new `Value` representing the result of the addition operation.
    ///
    /// - Example:
    ///   ```swift
    ///   let doubleValue = 4.0
    ///   let value = Value(6.0)
    ///   let result = doubleValue + value
    ///   ```
    ///   In this example, `result` will be a new `Value` representing the sum of 4.0 and 6.0.
    static func + (lhs: Double, rhs: Value) -> Value {
        return Value(lhs) + rhs
    }
    
}

// Used to get protocol-based support for Numeric Variables and Error Values
extension Value: FloatingPoint {
    convenience init(sign: FloatingPointSign, exponent: Int, significand: Value) {
        // Assuming data is of type Double
        let value = Double(sign: sign, exponent: exponent, significand: significand.data)
        self.init(value)
    }
        // Implement other required properties/methods of FloatingPoint
        var exponentBitPattern: UInt { return data.exponentBitPattern }
    var significandBitPattern: UInt { return UInt(data.significandBitPattern) }


    var exponent: Int {
        // Assuming data is of type Double
        return Int(data.exponent)
    }

    func distance(to other: Value) -> Value {
        // Assuming data is of type Double
        let distanceValue = Value(data.distance(to: other.data))
        return distanceValue
    }

    func advanced(by n: Value) -> Self {
        // Assuming data is of type Double
        Self(data.advanced(by: n.data))
    }
    
    typealias Exponent = Int
    
    convenience init(signOf: Value, magnitudeOf: Value) {
        // Assuming data is of type Double
        let signedMagnitude = Foundation.copysign(magnitudeOf.data, signOf.data)
        self.init(signedMagnitude)
    }

    
    convenience init<Source>(_ value: Source) where Source: BinaryInteger {
        // Assuming data is of type Double
        self.init(Double(value))
    }

    
    static var radix: Int {
        2
    }
    
    static var nan: Value {
        Value(.nan)
    }
    
    static var signalingNaN: Value {
        Value(.signalingNaN)
    }
    
    static var infinity: Value {
        Value(.infinity)
    }
    
    static var greatestFiniteMagnitude: Value {
        Value(.greatestFiniteMagnitude)
    }
    
    static var pi: Value {
        Value(.pi)
    }
    
    var ulp: Value {
        Value(data.ulp)
    }
    
    static var leastNormalMagnitude: Value {
        Value(.leastNormalMagnitude)
    }
    
    static var leastNonzeroMagnitude: Value {
        Value(.leastNonzeroMagnitude)
    }
    
    var sign: FloatingPointSign {
        data.sign
    }
    
    var significand: Value {
        Value(data.significand)
    }
    
    func formRemainder(dividingBy other: Value) {
           self.remainder(dividingBy: other)
        }

        func formTruncatingRemainder(dividingBy other: Value) {
            self.truncatingRemainder(dividingBy: other)
        }
    
    func formSquareRoot() {
        // Calculate the result of the square root operation.
        let sqrtResult = Value(Foundation.sqrt(self.data), children: [self], operations: "sqrt")
        
        // Define the backward closure for gradient calculation during backpropagation.
        sqrtResult._backward = {
            self.grad += 0.5 / sqrtResult.data * sqrtResult.grad
        }
        
        // Update the value with the result of the square root operation.
        self.data = sqrtResult.data
        
    }
    
    func addProduct(_ lhs: Value, _ rhs: Value) {
        var result = ( lhs * rhs)
        result = self + result
        
        self.data = result.data
        self._backward = result._backward
        self.children = result.children
    }
    
    var nextUp: Value {
        Value(data.nextUp)
    }
    
    func isEqual(to other: Value) -> Bool {
        data.isEqual(to: other.data)
    }
    
    func isLess(than other: Value) -> Bool {
        data.isLess(than: other.data)
    }
    
    func isLessThanOrEqualTo(_ other: Value) -> Bool {
        data.isLessThanOrEqualTo(other.data)
    }
    
    func isTotallyOrdered(belowOrEqualTo other: Value) -> Bool {
        data.isTotallyOrdered(belowOrEqualTo: other.data)

    }
    
    var isNormal: Bool {
        data.isNormal
    }
    
    var isFinite: Bool {
        data.isFinite
    }
    
    var isZero: Bool {
        data.isZero
    }
    
    var isSubnormal: Bool {
        data.isSubnormal
    }
    
    var isInfinite: Bool {
        data.isInfinite
    }
    
    var isNaN: Bool {
        data.isNaN
    }
    
    var isSignalingNaN: Bool {
        data.isSignalingNaN
    }
    
    var isCanonical: Bool {
        data.isCanonical
    }
    
    typealias Stride = Value
    
    convenience init(integerLiteral value: Int) {
        self.init(Double(value))
    }
    
    var magnitude: Double {
        return Swift.abs(data)
    }
    
    /// Initializes a Value with an integer literal.
    convenience init(_ value: Int) {
        self.init(Double(value))
    }
    
    typealias Magnitude = Double
    
    /// Initializes a Value with a binary integer.
    convenience init?<T>(exactly source: T) where T: BinaryInteger {
        self.init(Double(source))
    }
    
    typealias IntegerLiteralType = Int
    
    /// Performs subtraction of two `Value` instances.
    ///
    /// This operator calculates the result of subtracting the right-hand `Value` from the left-hand `Value`.
    ///
    /// - Parameters:
    ///   - lhs: The minuend `Value`.
    ///   - rhs: The subtrahend `Value`.
    ///
    /// - Returns: A new `Value` representing the result of the subtraction operation.
    ///
    /// - Example:
    ///   ```swift
    ///   let minuend = Value(5.0)
    ///   let subtrahend = Value(3.0)
    ///   let result = minuend - subtrahend
    ///   ```
    ///   In this example, `result` will be a new `Value` representing the result of subtracting 3.0 from 5.0.
    static func - (lhs: Value, rhs: Value) -> Value {
        return lhs + (-1 * rhs)
    }
    
    /// Performs subtraction of a `Value` by a `Double`.
    ///
    /// This operator calculates the result of subtracting a `Double` from a `Value`.
    ///
    /// - Parameters:
    ///   - lhs: The minuend `Value`.
    ///   - rhs: The subtrahend `Double`.
    ///
    /// - Returns: A new `Value` representing the result of the subtraction operation.
    ///
    /// - Example:
    ///   ```swift
    ///   let minuend = Value(10.0)
    ///   let subtrahend = 2.0
    ///   let result = minuend - subtrahend
    ///   ```
    ///   In this example, `result` will be a new `Value` representing the result of subtracting 2.0 from 10.0.
    static func - (lhs: Value, rhs: Double) -> Value {
        return lhs - Value(rhs)
    }
    
    /// Performs subtraction of a `Double` from a `Value`.
    ///
    /// This operator calculates the result of subtracting a `Value` from a `Double`.
    ///
    /// - Parameters:
    ///   - lhs: The minuend `Double`.
    ///   - rhs: The subtrahend `Value`.
    ///
    /// - Returns: A new `Value` representing the result of the subtraction operation.
    ///
    /// - Example:
    ///   ```swift
    ///   let minuend = 8.0
    ///   let subtrahend = Value(3.0)
    ///   let result = minuend - subtrahend
    ///   ```
    ///   In this example, `result` will be a new `Value` representing the result of subtracting 3.0 from 8.0.
    static func - (lhs: Double, rhs: Value) -> Value {
        return Value(lhs) - rhs
    }
    
    /// Performs division of two `Value` instances.
    ///
    /// This operator calculates the result of dividing the left-hand `Value` by the right-hand `Value`.
    ///
    /// - Parameters:
    ///   - lhs: The numerator `Value`.
    ///   - rhs: The denominator `Value`.
    ///
    /// - Returns: A new `Value` representing the result of the division operation.
    ///
    /// - Note: The division is achieved by multiplying the numerator by the reciprocal of the denominator.
    ///   If the denominator is zero, a warning is printed, and the result is set to `Double.nan`.
    ///
    /// - Example:
    ///   ```swift
    ///   let numerator = Value(6.0)
    ///   let denominator = Value(3.0)
    ///   let result = numerator / denominator
    ///   ```
    ///   In this example, `result` will be a new `Value` representing the result of 6.0 divided by 3.0.
    
    static func / (lhs: Value, rhs: Value) -> Value {
        if rhs.data == 0 {
            print("Warning: Division by zero.")
            return Value(Double.nan)
        }
        return lhs * (rhs ** (-1))
    }
    
    /// Performs division of a `Value` by a `Double`.
    ///
    /// This operator calculates the result of dividing a `Value` by a `Double`.
    ///
    /// - Parameters:
    ///   - lhs: The numerator `Value`.
    ///   - rhs: The denominator `Double`.
    ///
    /// - Returns: A new `Value` representing the result of the division operation.
    ///
    /// - Note: If the denominator is zero, a warning is printed, and the result is set to `Double.nan`.
    ///
    /// - Example:
    ///   ```swift
    ///   let numerator = Value(10.0)
    ///   let denominator = 2.0
    ///   let result = numerator / denominator
    ///   ```
    ///   In this example, `result` will be a new `Value` representing the result of 10.0 divided by 2.0.
    
    static func / (lhs: Value, rhs: Double) -> Value {
        if rhs == 0 {
            print("Warning: Division by zero.")
            return Value(Double.nan)
        }
        return lhs * (1 / rhs)
    }
    
    /// Performs division of a `Double` by a `Value`.
    ///
    /// This operator calculates the result of dividing a `Double` by a `Value`.
    ///
    /// - Parameters:
    ///   - lhs: The numerator `Double`.
    ///   - rhs: The denominator `Value`.
    ///
    /// - Returns: A new `Value` representing the result of the division operation.
    ///
    /// - Note: If the denominator is zero, a warning is printed, and the result is set to `Double.nan`.
    ///
    /// - Example:
    ///   ```swift
    ///   let numerator = 5.0
    ///   let denominator = Value(2.0)
    ///   let result = numerator / denominator
    ///   ```
    ///   In this example, `result` will be a new `Value` representing the result of 5.0 divided by 2.0.
    
    static func / (lhs: Double, rhs: Value) -> Value {
        if rhs.data == 0 {
            print("Warning: Division by zero.")
            return Value(Double.nan)
        }
        return lhs * (rhs ** (-1))
    }
    
    
    
    /// Performs multiplication of two `Value` instances.
    ///
    /// This operator calculates the result of multiplying two `Value` instances and sets up the backward closure
    /// for gradient calculation during backpropagation.
    ///
    /// - Parameters:
    ///   - lhs: The left-hand operand `Value`.
    ///   - rhs: The right-hand operand `Value`.
    ///
    /// - Returns: A new `Value` representing the result of the multiplication operation.
    ///
    /// - Note: The `operations` property of the resulting `Value` is set to "*" to indicate the operation performed.
    ///
    /// - Example:
    ///   ```swift
    ///   let value1 = Value(2.0)
    ///   let value2 = Value(3.0)
    ///   let result = value1 * value2
    ///   ```
    ///   In this example, `result` will be a new `Value` representing the product of `value1` and `value2`.
    static func * (lhs: Value, rhs: Value) -> Value {
        // Calculate the result of the multiplication operation.
        let out = Value(lhs.data * rhs.data, children: [lhs, rhs], operations: "*")
        
        // Define the backward closure for gradient calculation during backpropagation.
        out._backward = {
            lhs.grad += rhs.data * out.grad
            rhs.grad += lhs.data * out.grad
        }
        
        // Return the result of the multiplication operation.
        return out
    }
    
    /// Performs multiplication of a `Value` and a `Double`.
    ///
    /// This operator calculates the result of multiplying a `Value` instance by a `Double`.
    ///
    /// - Parameters:
    ///   - lhs: The left-hand operand `Value`.
    ///   - rhs: The right-hand operand `Double`.
    ///
    /// - Returns: A new `Value` representing the result of the multiplication operation.
    ///
    /// - Example:
    ///   ```swift
    ///   let value = Value(2.0)
    ///   let doubleValue = 3.0
    ///   let result = value * doubleValue
    ///   ```
    ///   In this example, `result` will be a new `Value` representing the product of `value` and `doubleValue`.
    static func * (lhs: Value, rhs: Double) -> Value {
        return lhs * Value(rhs)
    }
    
    /// Performs multiplication of a `Double` and a `Value`.
    ///
    /// This operator calculates the result of multiplying a `Double` by a `Value` instance.
    ///
    /// - Parameters:
    ///   - lhs: The left-hand operand `Double`.
    ///   - rhs: The right-hand operand `Value`.
    ///
    /// - Returns: A new `Value` representing the result of the multiplication operation.
    ///
    /// - Example:
    ///   ```swift
    ///   let doubleValue = 2.0
    ///   let value = Value(3.0)
    ///   let result = doubleValue * value
    ///   ```
    ///   In this example, `result` will be a new `Value` representing the product of `doubleValue` and `value`.
    static func * (lhs: Double, rhs: Value) -> Value {
        return Value(lhs) * rhs
    }
    
    /// Performs compound assignment multiplication of two `Value` instances.
    ///
    /// This operator calculates the result of multiplying two `Value` instances and assigns the result to the left-hand operand.
    ///
    /// - Parameters:
    ///   - lhs: The left-hand operand `Value`.
    ///   - rhs: The right-hand operand `Value`.
    static func *= (lhs: inout Value, rhs: Value) {
        lhs = lhs * rhs
    }
    
    /// Performs compound assignment multiplication of a `Value` and a `Double`.
    ///
    /// This operator calculates the result of multiplying a `Value` instance by a `Double` and assigns the result to the left-hand operand.
    ///
    /// - Parameters:
    ///   - lhs: The left-hand operand `Value`.
    ///   - rhs: The right-hand operand `Double`.
    static func *= (lhs: inout Value, rhs: Double) {
        lhs = lhs * rhs
    }
    
    /// Performs an exponential operation on a `Value` with a `Double` exponent.
    ///
    /// This operator calculates the result of raising a `Value` instance to the power of a `Double` exponent.
    ///
    /// - Parameters:
    ///   - lhs: The base `Value` to be raised to the power of the exponent.
    ///   - rhs: The `Double` exponent.
    ///
    /// - Returns: A new `Value` representing the result of the exponential operation.
    ///
    /// - Note: The `operations` property of the resulting `Value` is set to "**" to indicate the operation performed.
    ///
    /// - Example:
    ///   ```swift
    ///   let baseValue = Value(2.0)
    ///   let exponent = 3.0
    ///   let result = baseValue ** exponent
    ///   ```
    ///   In this example, `result` will be a new `Value` representing 2.0 raised to the power of 3.0.
    static func ** (lhs: Value, rhs: Double) -> Value {
        // Calculate the result of the exponential operation.
        let out = Value(pow(lhs.data, rhs), children: [lhs], operations: "**")
        
        // Define the backward closure for gradient calculation during backpropagation.
        out._backward = {
            lhs.grad += rhs * pow(lhs.data, rhs - 1) * out.grad
        }
        
        // Return the result of the exponential operation.
        return out
    }
    
}

/// Applies the hyperbolic tangent (tanh) activation function to a `Value` instance.
///
/// The tanh activation function maps input values to the range [-1, 1], providing non-linearity to neural networks.
/// This method also computes the gradient of the tanh function during backpropagation.
///
/// - Returns: A new `Value` representing the result of applying the tanh activation function.
///
/// - Note: The `operations` property of the resulting `Value` is set to "tanh" to indicate the operation performed.
///
/// - Example:
///   ```swift
///   let inputValue = Value(0.5)
///   let activatedValue = inputValue.tanh()
///   ```
///   In this example, `activatedValue` will be a new `Value` representing the tanh activation of the input value.
extension Value {
    func tanh() -> Value {
        // Calculate the tanh activation function value.
        let val = exp(2.0 * self.data)
        let out = Value((val - 1) / (val + 1), children: [self], operations: "tanh")
        
        // Define the backward closure for gradient calculation during backpropagation.
        out._backward = {
            self.grad += (1 - pow(out.data, 2.0)) * out.grad
        }
        
        // Return the result of the tanh activation function.
        return out
    }
}


// Backward pass
/// Performs a backward pass using topological sort to create a Directed Acyclic Graph (DAG).
/// The backward pass is a crucial step in gradient-based optimization algorithms, where it calculates gradients
/// with respect to the input variables by traversing the computation graph in reverse order.
/// - Note: This method assumes that the `Value` instance represents a node in a computation graph.
///
/// - Complexity: O(V + E), where V is the number of vertices (nodes) and E is the number of edges in the graph.
///
/// - Parameters:
///   - self: The `Value` instance representing the current node in the computation graph.
///
/// - Returns: None
///
/// - Important: The `grad` property of the input `Value` is set to 1.0 before the backward pass starts.
///   The topological sort is used to ensure that the traversal is in the correct order to avoid cycles.
///
/// - SeeAlso: `_backward()`
extension Value {
    func backward() {
        // A list to store the topologically sorted nodes during traversal.
        var sorted: [Value] = []
        
        // A set to keep track of visited nodes to avoid redundant traversal.
        var visited: Set<Value> = []
        
        // Helper function to recursively build the topological sort.
        func buildTopologicalSort(_ vertex: Value) {
            if !visited.contains(vertex) {
                visited.insert(vertex)
                // Recursively traverse the children of the current node.
                for child in vertex.children {
                    buildTopologicalSort(child)
                }
                // Add the current node to the sorted list after traversing its children.
                sorted.append(vertex)
            }
        }
        
        // Start the topological sort from the current node.
        buildTopologicalSort(self)
        
        // Set the gradient for the current node to 1.0, indicating the start of the backward pass.
        self.grad = 1.0
        
        // Reverse the sorted list to perform the backward pass in the correct order.
        sorted.reverse()
        
        // Perform the backward pass by calling the `_backward()` method on each node.
        for value in sorted {
            value._backward()
        }
    }
}



