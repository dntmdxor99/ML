import sys,os
import numpy as np
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from Differential import numerical_gradient, numerical_diff
sys.path.append('..')
from Differential import numerical_gradient, numerical_diff


def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient.numerical_gradient(f, x)
        x -= lr * grad

    return x


if __name__ == "__main__":
    value = gradient_descent(numerical_diff.function_2, np.array([-3.0, 4.0]), lr = 0.1, step_num=100)
    print(value)