//
//  Tensort.swift
//  micrograd
//
//  Created by Kruthay Kumar Reddy Donapati on 1/24/24.
//

import Foundation

/// Represents a multi-dimensional tensor.
/// Represents a tensor with gradient values for each element.
class Tensor {
    
    // MARK: - Properties
    
    /// Gradient values for each element in the tensor.
    var grad: [Double] {
        return storage.map { $0.grad }
    }
    
    /// Shape of the tensor.
    private(set) var shape: [Int]
    
    /// Total number of elements in the tensor.
    let totalSize: Int
    
    /// Storage for the values of the tensor.
    var storage: [Value]
    
    /// Computes the linear index from the given multi-dimensional position.
    private lazy var linearIndex: ([Int]) -> Int = { [unowned self] position in
        assert(position.count == self.shape.count)
        return zip(self.shape, position).reduce(0) { accumulator, element in
            assert(element.1 >= 0 && element.1 < element.0)
            return accumulator * element.0 + element.1
        }
    }
    
    
    // MARK: - Initializers
    
    /// Initializes a tensor with a specified initial value and dimensions.
    ///
    /// - Parameters:
    ///   - initialValue: The initial `Value` for all elements in the tensor.
    ///   - dimensions: The dimensions of the tensor.
    init(initialValue: Value, dimensions: Int...) {
        self.shape = dimensions
        self.totalSize = dimensions.reduce(1, *)
        self.storage = Array(repeating: initialValue, count: totalSize)
    }
    
    /// Initializes a tensor with an array of values.
    ///
    /// - Parameter data: An array of `Value` instances representing the values of the tensor.
    init(_ data: [Value]) {
        self.shape = [data.count]
        self.totalSize = data.count
        self.storage = data
    }
    
    /// Initializes a tensor with a scalar value.
    ///
    /// - Parameter scalarValue: The scalar `Value` for the tensor.
    init(_ scalarValue: Value) {
        self.shape = [1]
        self.totalSize = 1
        self.storage = [scalarValue]
    }
}


extension Tensor {
    
    // MARK: - View
    
    /// Reshapes the tensor to the specified dimensions.
    ///
    /// - Parameter shape: The new dimensions.
    /// - Returns: `true` if the reshaping is successful, otherwise `false`.
    func view(to shape: Int...) -> Bool {
        let newTotalSize = shape.reduce(1, *)
        guard totalSize == newTotalSize else {
            print("Error: Incompatible dimensions")
            return false
        }
        
        self.shape = shape
        return true
    }
    
    // MARK: - Subscripts
    
    /// Accesses the element at the specified position in the tensor using variadic indices.
    ///
    /// - Parameter position: The indices specifying the position.
    ///
    /// - Returns: The `Value` at the specified position.
    subscript(position: Int...) -> Value {
        get {
            let linearIdx = linearIndex(position)
            assert(linearIdx < storage.count, "Index out of bounds")
            return storage[linearIdx]
        }
        set {
            let linearIdx = linearIndex(position)
            assert(linearIdx < storage.count, "Index out of bounds")
            storage[linearIdx] = newValue
        }
    }
    
    /// Accesses the element at the specified indices in the tensor.
    ///
    /// - Parameter indices: An array of indices specifying the position.
    ///
    /// - Returns: The `Value` at the specified position.
    subscript(indices: [Int]) -> Value {
        get {
            let linearIdx = linearIndex(indices)
            precondition(linearIdx < storage.count, "Index out of bounds")
            return storage[linearIdx]
        }
        set {
            let linearIdx = linearIndex(indices)
            precondition(linearIdx < storage.count, "Index out of bounds")
            storage[linearIdx] = newValue
        }
    }
    
    // MARK: - Helper Functions
    
    /// Generates a string representation of the tensor using recursion.
    ///
    /// - Parameters:
    ///   - tensor: The tensor to represent.
    ///   - indices: The current indices in the recursion.
    ///   - depth: The depth of recursion.
    ///
    /// - Returns: A string representation of the tensor.
    private func recursiveString(tensor: Tensor, indices: [Int], depth: Int) -> String {
        var result = ""
        if depth == tensor.shape.count {
            result += "\(tensor[indices])"
        } else {
            let currentDimension = tensor.shape[depth]
            result += "["
            for i in 0..<currentDimension {
                let newIndices = indices + [i]
                result += recursiveString(tensor: tensor, indices: newIndices, depth: depth + 1)
                if i < currentDimension - 1 {
                    result += ", "
                }
            }
            result += "]"
            if depth < tensor.shape.count - 1 {
                result += ","
            }
        }
        return result
    }
}


// MARK: - Extensions

extension Tensor: Sequence {
    
    /// Returns an iterator over the tensor's storage values.
    ///
    /// - Returns: An `IndexingIterator` over the `Value` instances in the tensor's storage.
    func makeIterator() -> IndexingIterator<[Value]> {
        return storage.makeIterator()
    }
}


extension Tensor: CustomStringConvertible {
    
    /// A string representation of the tensor.
    ///
    /// This property generates a human-readable string representation of the tensor's structure and values. It uses a recursive approach to represent multi-dimensional tensors in a nested format.
    ///
    /// - Note: This representation may be lengthy for large tensors.
    ///
    /// - Returns: A string describing the tensor.
    var description: String {
        return recursiveString(tensor: self, indices: [], depth: 0)
    }
}


// MARK: - Vector Extension

extension Tensor {
    
    // MARK: - Vector Operations
    
    /// Adds two tensors element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The left-hand tensor.
    ///   - rhs: The right-hand tensor.
    ///
    /// - Returns: A new tensor representing the element-wise addition of the two tensors.
    ///
    /// - Note: The tensors must have the same shape for this operation to be valid.
    static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
        guard lhs.shape == rhs.shape else {
            fatalError("Vector dimensions do not match for addition.")
        }
        
        let resultValues = zip(lhs.storage, rhs.storage).map { $0 + $1 }
        return Tensor(resultValues)
    }
    
    /// Multiplies two tensors element-wise.
    ///
    /// - Parameters:
    ///   - lhs: The left-hand tensor.
    ///   - rhs: The right-hand tensor.
    ///
    /// - Returns: A new tensor representing the element-wise multiplication of the two tensors.
    ///
    /// - Note: The tensors must have the same shape for this operation to be valid.
    static func * (lhs: Tensor, rhs: Tensor) -> Tensor {
        guard lhs.shape == rhs.shape else {
            fatalError("Vector dimensions do not match for multiplication.")
        }
        
        let resultValues = zip(lhs.storage, rhs.storage).map { $0 * $1 }
        return Tensor(resultValues)
    }
    
    // MARK: - Additional Vector Operations
    
    /// Performs the backward pass on the entire vector.
    ///
    /// - Note: This operation is valid only for tensors with a total size of 1 (scalars).
    func backward() {
        if self.totalSize != 1 {
            print("Error: backward can only be applied to a scalar.")
            return
        }
        
        self.storage.first!.backward()
    }
}
