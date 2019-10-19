import matplotlib.pyplot as plt
import numpy as np


def sse(p_x, p_y, predicted_y):
    return (1 / p_x.shape[0]) * (np.sum(pow(np.subtract(p_y, predicted_y), 2)))


def display_plot(m, n, x, y):
    plt.figure('Adaline Figure [with data]')
    px1, px2, py1, py2 = list(), list(), list(), list()

    for coord, out in zip(x, y):
        if out == -1 or out == 0:
            px1.append(coord[0])
            py1.append(coord[1])
        else:
            px2.append(coord[0])
            py2.append(coord[1])

    plt.plot(px1, py1, 'o', c='y')
    plt.plot(px2, py2, 'o', c='r')
    vec_x = np.linspace(-1, 8, 100)
    vec_y = np.multiply(m, vec_x) + n
    plt.plot(vec_x, vec_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.title('NN Progress')
    plt.show()


def display_error(nn):
    plt.figure('SSE Progress')
    plt.plot(list(range(0, nn.get_epochs())), nn.get_errors())
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True)
    plt.title('Epoch VS Error')
    plt.show()
