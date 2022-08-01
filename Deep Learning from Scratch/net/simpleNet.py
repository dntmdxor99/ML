import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from common.Activation_Function.Softmax_Function import softmax
from common.Loss_Function.cross_entropy_error import cross_entropy
from common.Differential.numerical_gradient import numerical_gradient_2d


class simpleNet:
    # 간단한 네트워크
    def __init__(self) -> None:
        self.W = np.random.randn(2, 3)      # 정규 분포로 초기화


    def predict(self, x):
        # 해당 입력 값을 바탕으로 가중치를 곱하여 예측함
        return np.dot(x, self.W)        # 가중치와 입력 값을 곱함


    def loss(self, x, t):
        # 입력 값에 해당하는 예측 확률과 target 값을 바탕으로 교차 엔트로피를 수행함
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy(y, t)

        return loss


if __name__ == "__main__":
    net = simpleNet()
    print(net.W)

    x = np.array([0.6, 0.9])

    p = net.predict(x)
    print(p)

    t = np.array([0, 0, 1])
    loss = net.loss(x, t)
    print(loss)

    
    def f(W):
        return net.loss(x, t)

    
    dW = numerical_gradient_2d(f, net.W)
    print(dW)