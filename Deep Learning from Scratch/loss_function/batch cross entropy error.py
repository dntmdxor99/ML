import numpy as np
from mnist import load_mnist


def batch_cross_entropy_error(pred, target):
    if pred.ndim == 1:
        target = target.reshape(1, target.size)
        pred = pred.reshape(1, pred.size)

    batch_size = pred.shape[0]
    return -np.sum(target * np.log(pred + 1e-7)) / batch_size
    # return -np.sum(np.log(target[np.arange(batch_size), target] + 1e-7)) / batch_size


if __name__ == "__main__":
    (input_train, target_train), (input_test, target_test) = load_mnist(normalize=True, one_hot_label=True)

    train_size = input_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)

    input_batch = input_train[batch_mask]
    target_batch = target_train[batch_mask]
    pred_batch = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    # pred_batch = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    error = batch_cross_entropy_error(pred_batch, target_batch)

    print(error)
