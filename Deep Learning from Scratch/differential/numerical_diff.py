def numerical_diff(f, x):
    h = 1e-4
    return ((f(x + h) - f(x - h)) / (2 * h))


def function_1(x):
    # 수치 미분용 함수
    return 0.01 * x ** 2 + 0.1 * x


def function_2(x):
    # 편 미분용 함수
    return x[0] ** 2 + x[1] ** 2


if __name__ == "__main__":
    value = numerical_diff(function_1, 5)
    print(value)