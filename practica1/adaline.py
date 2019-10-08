import numpy as np
from utils import displayPlot
import sys

class Adaline:
    def __init__(self, eta=0.01, epochs=50, random_state=1):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.errors = list()

    def fit(self, p_x, p_y, showProgress=-1):
        random_seed = np.random.RandomState(seed=self.random_state)
        self.w = random_seed.normal(scale=0.01, size= 1 + p_x.shape[1])
        for epoch in range(self.epochs):
            output = self.activation(self.net_input(p_x))
            gradient = self.calculate_gradient(output, p_y, p_x)
            self.w = np.add(self.w, self.calculate_weight(gradient))
            self.errors.append(self.sse(p_x, p_y, output))

            if showProgress != -1 and epoch % showProgress == 0:
                self.plot(p_x, p_y)

    def net_input(self, p_x):
        return np.dot(p_x, self.w[1:]) + self.w[0]

    def activation(self, p_net_input):
        return p_net_input
    
    def quantization(self, p_x):
        return np.where(p_x < 0, -1, 1)

    def predict(self, p_x):
        return self.quantization(self.activation(self.net_input(p_x)))

    def calculate_gradient(self, predicted_y, p_y, p_x):
        xj = np.append(np.ones((p_x.shape[0], 1)), p_x, axis=1)
        return xj.T.dot((p_y - predicted_y))

    def calculate_weight(self, gradient):
        return self.eta * gradient

    def plot(self, x, y):
        displayPlot(- self.w[1]/self.w[2], - self.w[0]/self.w[2], x, y)
    
    def get_weights(self):
        return self.w

    def sse(self, p_x, p_y, predicted_y):
        return (1/p_x.shape[0])*(np.sum(pow(np.subtract(p_y, predicted_y),2)))

    def get_errors(self):
        return self.errors

    def get_epochs(self):
        return self.epochs