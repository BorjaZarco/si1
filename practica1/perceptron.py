from utils import display_plot, sse
import numpy as np


class Perceptron:
    def __init__(self, eta=0.1, epochs=10):
        self.eta = eta
        self.epochs = epochs
        self.errors = list()
        self.w = None

    def fit(self, p_x, p_y, show_progress=-1):
        p_y_fixed = [-1 if i == 0 else 1 for i in p_y]
        self.w = np.ones(1 + p_x.shape[1])
        out = None
        for epoch in range(self.epochs):
            for x, y in zip(p_x, p_y):
                out = self.generate_output(x)
                diff_w = self.eta * (y - out) * x
                self.w[1:] = np.add(self.w[1:], diff_w)
                self.w[0] = self.w[0] + self.eta * (y - out)

            self.errors.append(sse(p_x, p_y_fixed, out))
            if show_progress != -1 and (epoch + 1) % show_progress == 0:
                self.plot(p_x, p_y)

    def predict(self, input_vector):
        return [self.generate_output(x) for x in input_vector]

    def generate_output(self, vector_x):
        return int(np.sum(np.multiply(vector_x, self.w[1:])) > self.w[0])

    def plot(self, x, y):
        display_plot(- self.w[1] / self.w[2], self.w[0] / self.w[2], x, y)

    def get_weights(self):
        return self.w

    def get_errors(self):
        return self.errors

    def get_epochs(self):
        return self.epochs
