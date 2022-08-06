from audioop import cross
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common import *
import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01) -> None:
        # 파라미터를 정규 분포로 초기화 함
        self.params = {}

        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


    def predict(self, x : np.array) -> np.array:
        # 신경망의 파라미터와 x 값을 행렬곱, 행렬덧셈을 하여 sigmoid, softmax를 통해 예상 값(확률)을 내놓음

        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y


    def loss(self, x : np.array, t : np.array) -> np.array:
        # 예상한 값을 바탕으로 cross_entropy를 수행하여 손실 함수 값을 구함

        y = self.predict(x)

        return cross_entropy(y, t)


    def accuracy(self, x : np.array, t : np.array) -> float:
        # 예상한 값에서 가장 확률이 높은 index -> A
        # 타겟 값에서 정답인 index -> B
        # A와 B를 비교하여 정확도를 계산함
        
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    
    def numerical_gradient(self, x : np.array, t : np.array) -> dict:
        # 신경망의 파라미터가 손실 함수에 대하여 얼만큼의 기울기를 가지고 있는지 계산함
        # 쉽게 말하여 파라미터가 얼마나 손실함수의 증감에 영향을 끼치는지를 파악함

        loss_W = lambda W : self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient_2d(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient_2d(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient_2d(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient_2d(loss_W, self.params['b2'])

        return grads


if __name__ == "__main__":
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    x = np.random.randn(100, 784)
    y = net.predict(x)

    print(y)