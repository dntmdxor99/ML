import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1


for i in [0, 1]:
    for j in [0, 1]:
        print(f'{i, j}', end=' ')

print()
print('AND : ', end = '')
for i in [0, 1]:
    for j in [0, 1]:
        print(AND(i, j), end = ' ')

print()
print('NAND : ', end = '')
for i in [0, 1]:
    for j in [0, 1]:
        print(NAND(i, j), end = ' ')

print()
print('OR : ', end = '')
for i in [0, 1]:
    for j in [0, 1]:
        print(OR(i, j), end = ' ')

