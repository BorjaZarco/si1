import numpy as np
from utils import displayPlot
import sys

class Perceptron:
    def __init__(self, eta=0.1, epochs=10):
        self.eta = eta
        self.epochs = epochs

    def fit(self, X, Y, showProgress=-1):
        self.w = np.ones(1 + X.shape[1])
        for i in range(self.epochs):
            for x,y in zip(X,Y):
                out = self.generate_output(x)
                Δw = self.eta * (y - out) * x
                self.w[1:] = np.add(self.w[1:], Δw)
                self.w[0] = self.w[0] + self.eta * (y - out)
            if showProgress != -1 and i % showProgress == 0:
                self.plot(X, Y)
    
    def predict(self, input_vector):
        return [ self.generate_output(x) for x in input_vector]

    def generate_output(self, vector_x):
        return int(np.sum(np.multiply(vector_x, self.w[1:])) > self.w[0])

    def plot(self, x, y):
        displayPlot(-self.w[1]/self.w[2], self.w[0]/self.w[2], x, y)
    
    def get_weights(self):
        return self.w