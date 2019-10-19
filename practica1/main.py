from perceptron import Perceptron
from adaline import Adaline
from utils import display_error
import numpy as np
import pandas as pd


def nn_process(nn, p_x, p_y):
    nn.fit(p_x, p_y, show_progress=500)
    prediction = nn.predict(p_x)
    accuracy = len([1 for predicted, expected in zip(prediction, p_y) if predicted == expected])
    display_error(nn)
    print("Accuracy: ", round(accuracy * 100 / len(p_y), 2), "%")
    print("Pesos Finales: ", nn.get_weights())


def own_test_basic(p_x, p_y):
    print("============|Own Test begins... [Basic]|===========")
    perceptron = Perceptron(eta=0.001, epochs=100)
    nn_process(perceptron, p_x, p_y)
    print("=============|End of Own Test [Basic]|=============\n")


def own_test_adaline(p_x, p_y):
    print("============|Own Test begins... [Adaline]|===========")
    adaline = Adaline(eta=0.001, epochs=1000)
    nn_process(adaline, p_x, p_y)
    print("=============|End of Own Test [Adaline]|=============\n")


def iris_test_basic_setosa_versicolor(p_x, p_y):
    print("============|Iris Test begins... [Basic]|===========")
    perceptron = Perceptron(eta=0.0001, epochs=3000)
    nn_process(perceptron, p_x, p_y)
    print("=============|End of Iris Test [Basic]|=============\n")


def iris_test_adaline_setosa_versicolor(p_x, p_y):
    print("============|Iris Test begins... [Adaline]|===========")
    adaline = Adaline(eta=0.001, epochs=1000)
    nn_process(adaline, p_x, p_y)
    print("=============|End of Iris Test [Adaline]|=============\n")


def iris_test_basic_versicolor_virginica(p_x, p_y):
    print("============|Iris Test begins... [Basic]|===========")
    perceptron = Perceptron(eta=0.0001, epochs=3000)
    nn_process(perceptron, p_x, p_y)
    print("=============|End of Iris Test [Basic]|=============\n")


def iris_test_adaline_versicolor_virginica(p_x, p_y):
    print("============|Iris Test begins... [Adaline]|===========")
    adaline = Adaline(eta=0.0005, epochs=4000)
    nn_process(adaline, p_x, p_y)
    print("=============|End of Iris Test [Adaline]|=============\n")


if __name__ == "__main__":
    x = np.array(
        [[0, 3], [3, 0], [0, 5], [5, 3], [0, 0], [6, 4], [4, 6], [2, 2], [1, 1], [6, 6], [5, 5], [7, 2], [4, 6.5],
         [4, 2.5], [4.8, 0.25]])
    y_0 = np.array([0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0])
    y = np.array([-1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1])
    own_test_basic(x, y_0)
    own_test_adaline(x, y)

    iris_dataset = pd.read_csv("dataset/iris.csv")
    x_pre = list()
    for i in range(0, 100):
        x_pre.append([iris_dataset['petal.length'][i], iris_dataset['petal.width'][i]])
    x = np.array(x_pre)
    y_0 = np.concatenate((np.zeros(50), np.ones(50)))
    y = np.concatenate((-1 * np.ones(50), np.ones(50)))
    iris_test_basic_setosa_versicolor(x, y_0)
    iris_test_adaline_setosa_versicolor(x, y)

    x_pre = list()
    for i in range(50, 150):
        x_pre.append([iris_dataset['petal.length'][i], iris_dataset['petal.width'][i]])
    x = np.array(x_pre)
    iris_test_basic_versicolor_virginica(x, y_0)
    iris_test_adaline_versicolor_virginica(x, y)
