import numpy as np

class SGD:
    def __init__(self, lr : float = 0.01) -> None:
        self.lr = lr

    def update(self, params : dict, grads : dict):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum(SGD):
    def __init__(self, lr: float = 0.01, momentum : float = 0.9) -> None:
        super().__init__(lr)
        self.momentum = momentum
        self.v = None

    def update(self, params: dict, grads: dict):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

            for key in params.keys():
                self.v[key] = self.momentum * self.v[key] - grads[key]
                params[key] += self.v[key]


class AdaGrad(SGD):
    def __init__(self, lr: float = 0.01) -> None:
        super().__init__(lr)
        self.h = None

    def update(self, params: dict, grads: dict):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

            for key in params.keys():
                self.h[key] += grads[key] * grads[key]
                grads[key] -= self.lr * grads[key] / (np.sqrt(self.h[key] + 1e-7))


d