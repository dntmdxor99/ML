class MulLayer:
    # 곱셈 계층

    def __init__(self) -> None:
        self.x = None
        self.y = None


    def forward(self, x, y):
        # 순전파, x와 y의 값을 저장해야만 backward때 사용할 수 있다.
        self.x = x
        self.y = y
        out = x * y

        return out

    
    def backward(self, dout):
        # 역전파로 상위 계층에서의 미분 값 * 반대 노드의 값을 출력한다.
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    # 덧셈 계층

    def __init__(self) -> None:
        pass


    def forward(self, x, y):
        # 순전파, x와 y 값을 저장하지 않아도 된다.
        out = x + y
        return out


    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy