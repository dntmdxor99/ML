import numpy as np


def sum_squares_error(pred, target):
    # 오차제곱합
    return 0.5 * np.sum((pred - target) ** 2)


def cross_entropy(pred, target):
    # 교차 엔트로피 합
    delta = 1e-7        # log 0을 계산할 수 없으므로, 아주 작은 값을 넣음
    return -np.sum(target * np.log(pred + delta))


if __name__== "__main__":
    target = np.array([0, 0, 1, 0, 0])      # 2가 정답
    pred1 = np.array([0.1, 0.05, 0.6, 0.05, 0.2])       # 2 예측
    pred2 = np.array([0.6, 0.1, 0.05, 0.2, 0.05])       # 0 예측

    for func in [sum_squares_error, cross_entropy]:
        print(func.__name__)

        error1 = func(pred1, target)
        error2 = func(pred2, target)

        print(error1)
        print(error2)
