import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from common.Activation_Function.Softmax_Function import softmax
from common.Loss_Function.cross_entropy_error import cross_entropy
from common.Differential.numerical_gradient import numerical_gradient


class simpleNet:
    def __init__(self) -> None:
        self.W = np.random.randn(2, 3)      # 정규 분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)        # 가중치와 입력 값을 곱함

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy(y, t)

        return loss


net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])

p = net.predict(x)
print(p)

t = np.array([0, 0, 1])
loss = net.loss(x, t)
print(loss)

