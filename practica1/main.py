from perceptron import Perceptron
from utils import displayPlot
from sklearn import datasets
from pandas.plotting import scatter_matrix
import pandas as pd
import numpy as np

def own_test():
    print("Own Test begins...")
    x = np.array([[0,3],[3,0],[0,5],[5,3],[0,0],[6,4],[4,6],[2,2],[1,1],[6,6],[5,5],[7,2],[4,6.5],[4,2.5], [4.8,0.25]])
    y = np.array([0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0])
    perceptron = Perceptron(eta=0.15, epochs=100)
    perceptron.fit(x, y, showProgress=50)
    prediction = perceptron.predict(x)
    accuracy = len([1 for predicted, expected in zip(prediction, y) if predicted == expected ])
    print("Accuracy: ", round(accuracy*100/len(y),2), "%")
    print(perceptron.get_weights())

# def iris_test():
    # print("Iris Test begins...")
    # iris = datasets.load_iris()
    # x = iris.data[:, :2]  # we only take the first two features.
    # y = iris.target
    # print("xses: ",x," yses:",y)
    # perceptron = Perceptron(eta=0.15, epochs=100)
    # perceptron.fit(x, y, showProgress=50)
    # prediction = perceptron.predict(x)
    # accuracy = len([1 for predicted, expected in zip(prediction, y) if predicted == expected ])
    # print("Accuracy: ", round(accuracy*100/len(y),2), "%")
    # print(perceptron.get_weights())

if __name__ == "__main__":
    own_test()
    # iris_test()