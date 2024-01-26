//
//  main.swift
//  micrograd
//
//  Created by Kruthay Kumar Reddy Donapati on 1/23/24.
//

import Foundation


// Sample Example of MLP using Value
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
print(loss.data)

print(mlp.parameters.count)


loss.backward()

print(loss)

// Sample Implementation using Tensor

var x = Tensor([1,2,3])

var y = Tensor([5,6,7])

var z = Tensor([11,12,13])

var i = x + y

var j = i * z


j.reduce(Value(0)) { $0 + $1 }.backward()



print(x.grad)

print(z.description)
