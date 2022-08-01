import numpy as np


def softmax(a):
    c = np.mean(a)
    exp_a = np.exp(a - c)       # 오버 플로를 막기 위해 사용
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


if __name__ == "__main__":
    x = np.array([0.3, 2.9, 4.0])
    y = softmax(x)

    print(y)
