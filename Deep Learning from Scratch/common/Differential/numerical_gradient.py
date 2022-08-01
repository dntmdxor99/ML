import numpy as np


def numerical_gradient(f, x):
    # 함수 f를 x 좌표에서의 편미분 값을 반환하는 함수
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


def numerical_gradient_2d(f, x):
    if x.ndim == 1:
        return numerical_gradient(f, x)
    else:
        grad = np.zeros_like(x)
        
        for idx, x in enumerate(x):
            grad[idx] = numerical_gradient(f, x)
        
        return grad


if __name__ == "__main__":
    from numerical_diff import function_2
    value = numerical_gradient(function_2, np.array([3.0, 4.0]))
    print(value)