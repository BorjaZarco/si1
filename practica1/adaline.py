from utils import display_plot, sse
import numpy as np


def activation(p_net_input):
    return p_net_input


def quantization(output):
    return np.where(output < 0, -1, 1)


def calculate_gradient(predicted_y, p_y, p_x):
    xj = np.append(np.ones((p_x.shape[0], 1)), p_x, axis=1)
    return xj.T.dot((p_y - predicted_y))


class Adaline:
    def __init__(self, eta=0.01, epochs=50, random_state=1):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.errors = list()
        self.w = None

    def fit(self, p_x, p_y, show_progress=-1):
        random_seed = np.random.RandomState(seed=self.random_state)
        self.w = random_seed.normal(scale=0.01, size=1 + p_x.shape[1])
        for epoch in range(self.epochs):
            output = activation(self.net_input(p_x))
            gradient = calculate_gradient(output, p_y, p_x)
            self.w = np.add(self.w, self.calculate_weight(gradient))

            self.errors.append(sse(p_x, p_y, output))
            if show_progress != -1 and (epoch + 1) % show_progress == 0:
                self.plot(p_x, p_y)

    def net_input(self, p_x):
        return np.dot(p_x, self.w[1:]) + self.w[0]

    def predict(self, p_x):
        return quantization(activation(self.net_input(p_x)))

    def calculate_weight(self, gradient):
        return self.eta * gradient

    def plot(self, x, y):
        display_plot(- self.w[1] / self.w[2], - self.w[0] / self.w[2], x, y)

    def get_weights(self):
        return self.w

    def get_errors(self):
        return self.errors

    def get_epochs(self):
        return self.epochs
