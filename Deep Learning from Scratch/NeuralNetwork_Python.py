import numpy as np

class NeuralNetwork:
    def __init__(self, inputnodes : int, hiddennodes : int, outputnodes : int, learningrate : float) -> None:
        self.inodes = inputnodes        # input 노드 개수
        self.hnodes = hiddennodes       # hidden 노드 개수
        self.onodes = outputnodes       # output 노드 개수

        # 가중치 행렬을 설정 whi(weight hidden input), woh(weight output hidden)
        # whi[i][j] = j번째 노드에서 i번째 노드로 가는 가중치

        # self.whi = np.random.rand(self.hnodes, self.inodes) - 0.5
        # self.woh = np.random.rand(self.onodes, self.hnodes) - 0.5

        # Xavier 초기화 - > 1/sqrt(n) n : 이전 노드 개수

        self.whi = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.woh = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate      # 학습률
        pass


    def train(self, train_input, train_target):
        # 가중치를 학습하는 과정
        # input_T = np.array(input_list, ndmin = 2).T     # 입력 리스트를 2차원 행렬로 변환
        # target_T = np.array(input_list, ndmin = 2).T

        # 오차 값 계산
        error = self.mean_squared_error(train_input, train_target)


        pass


    def mean_squared_error(self, train_input, train_target):
        n = len(train_input)
        train_pred = self.predict(train_input)

        return 1/n * np.sum((train_pred - train_target) ** 2)


    def predict(self, test_input):
        # 들어온 값에 대하여 결과 값을 출력하는 함수
        test = np.array(test_input, ndmin = 2).T     # 2차원 배열로 만들고 Transpose 시켜줌
        
        # 가중치를 곱한 값을 sigmoid에 넣어 최종 값을 얻는 과정, sigmoid는 activation function임
        hidden_input = np.dot(self.whi, test)
        hidden_output = self.sigmoid(hidden_input)
        final_input = np.dot(self.woh, hidden_output)
        final_output = self.sigmoid(final_input)

        return final_output


    def sigmoid(self, x):
        # activation function
        return 1 / (1 + np.exp(-np.sum(x)))


if __name__ == "__main__":
    n = NeuralNetwork(3, 3, 3, 0.3)
    print(n.predict([1.0, -0.5, -1.5]))
