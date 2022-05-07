import sys, os
import numpy as np
import pickle
from PIL import Image
sys.path.append(os.pardir)
from mnist import load_mnist
from Activation_Function.func import sigmoid, softmax


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def show_img():
    # 첫 번째 이미지를 출력하는 함수
    (input_train, target_train), (input_test, target_test) = \
        load_mnist(flatten = True, normalize = False)


    img = input_train[0]
    label = target_train[0]
    print(label)

    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)

    img_show(img)


def get_data():
    (input_train, target_train), (input_test, target_test) = load_mnist(normalize = True, flatten = True, one_hot_label = False)
    return input_test, target_test


def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


if __name__ == "__main__":
    input, target = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0
    for i in range(0, len(input), batch_size):
        input_batch = input[i : i + batch_size]
        pred_batch = predict(network, input_batch)
        p = np.argmax(pred_batch, axis = 1)
        # print(p == target[i : i + batch_size])
        accuracy_cnt += np.sum(p == target[i : i + batch_size])

    print(f'Accuracy : {accuracy_cnt / len(input)}')

