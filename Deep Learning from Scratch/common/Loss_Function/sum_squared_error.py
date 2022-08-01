import numpy as np


def sum_squares_error(pred, target):
    # 오차제곱합
    return 0.5 * np.sum((pred - target) ** 2)


if __name__== "__main__":
    target = np.array([0, 0, 1, 0, 0])      # 2가 정답
    pred1 = np.array([0.1, 0.05, 0.6, 0.05, 0.2])       # 2 예측
    pred2 = np.array([0.6, 0.1, 0.05, 0.2, 0.05])       # 0 예측

    for func in [sum_squares_error]:
        print(func.__name__)

        error1 = func(pred1, target)
        error2 = func(pred2, target)

        print(error1)
        print(error2)
