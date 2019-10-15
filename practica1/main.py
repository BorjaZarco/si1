from adaline import Adaline
from perceptron import Perceptron
from utils import displayError
from sklearn import datasets
from pandas.plotting import scatter_matrix
import pandas as pd
import numpy as np

def own_test_basic(x, y):
    print("Own Test begins... [Basic]")
    perceptron = Perceptron(eta=0.015, epochs=250)
    perceptron.fit(x, y, showProgress=-1)
    prediction = perceptron.predict(x)
    accuracy = len([1 for predicted, expected in zip(prediction, y) if predicted == expected])
    print("Accuracy: ", round(accuracy*100/len(y),2), "%")
    print("Pesos Finales: ", perceptron.get_weights())
    displayError(perceptron)
    print("=============|End of Own Test [Basic]|=============")

def own_test_adaline(x, y):
    print("Own Test begins... [Adaline]")
    adaline = Adaline(eta=0.001, epochs=500)
    adaline.fit(x, y, showProgress=-1)
    prediction = adaline.predict(x)
    accuracy = len([1 for predicted, expected in zip(prediction, y) if predicted == expected])
    print("Accuracy: ", round(accuracy*100/len(y),2), "%")
    print("Pesos Finales: ", adaline.get_weights())
    displayError(adaline)
    print("=============|End of Own Test [Adaline]|=============")


if __name__ == "__main__":
    x = np.array([[0,3],[3,0],[0,5],[5,3],[0,0],[6,4],[4,6],[2,2],[1,1],[6,6],[5,5],[7,2],[4,6.5],[4,2.5], [4.8,0.25]])
    y = np.array([-1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1])
    y_0 = np.array([0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0])
    own_test_basic(x, y_0)
    own_test_adaline(x, y)
    # TODO: iris_test()