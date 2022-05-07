def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


for i in [0, 1]:
    for j in [0, 1]:
        print(f'{i, j}', end=' ')

print()
print('XOR : ', end = '')
for i in [0, 1]:
    for j in [0, 1]:
        print(XOR(i, j), end = ' ')
