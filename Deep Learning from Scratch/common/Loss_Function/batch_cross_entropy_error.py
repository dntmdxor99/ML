import os, sys

import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from mnist import load_mnist


def batch_cross_entropy_error(pred, target):
    
    if pred.ndim == 1:
        target = target.reshape(1, target.size)
        pred = pred.reshape(1, pred.size)

    batch_size = pred.shape[0]
    # return -np.sum(target * np.log(pred + 1e-7)) / batch_size     # 원-핫 인코딩 방식일 때
    return -np.sum(np.log(pred[np.arange(batch_size), target] + 1e-7)) / batch_size     # 레이블 방식일 때


if __name__ == "__main__":
    (input_train, target_train), (input_test, target_test) = load_mnist(normalize=True, one_hot_label=False)

    train_size = input_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size, replace = False) 

    input_batch = input_train[batch_mask]
    target_batch = target_train[batch_mask]
    pred_batch = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])

    error = batch_cross_entropy_error(pred_batch, target_batch)

    print(error)