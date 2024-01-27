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

loss.backward()

// Sample Implementation using Tensor

var x = Tensor([1,2,3])

var y = Tensor([5,6,7])
//x = Tensor([ 0.3583, -0.3830,  0.1439])
//
//y = Tensor([-0.3571, -1.1960,  0.3563])



x.view(to: 3,1)
y.view(to: 1,3)

if let m = x ^^ y {
    var s = m.tanh()
    var f = s.sum()
    print(f)
    f.backward()
    print(s)
}

print(x, y)

