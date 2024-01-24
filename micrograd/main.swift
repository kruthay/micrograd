//
//  main.swift
//  micrograd
//
//  Created by Kruthay Kumar Reddy Donapati on 1/23/24.
//

import Foundation
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
