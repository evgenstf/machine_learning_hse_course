import numpy as np
from sklearn.base import BaseEstimator
import random

class HuberReg(BaseEstimator):
    def __init__(
            self,
            delta = 1.0,
            gd_type = 'stochastic',
            tolerance = 1e-4,
            max_iter = 1000,
            w0 = None,
            alpha = 1e-3,
            eta = 1e-2
        ):
        self.delta = delta
        self.gradient_type = gd_type
        self.tolerance = tolerance
        self.max_iterations_count = max_iter
        self.alpha = alpha
        self.eta = eta
        self.loss_history = []
        self.w = None
        self.initial_weights = w0
    
    def fit(self, x_data, y_data):
        self.init_weights(x_data)
        self.update_loss_history(x_data, y_data)
        
        previous, current = 0, 0
        for iteration in range(self.max_iterations_count):
            if self.gradient_type == 'full':
                current = self.alpha * previous - self.eta * self.calc_gradient(x_data, y_data)
            elif self.gradient_type == 'stochastic':
                row_index = random.randint(0, np.shape(x_data)[0] - 1)
                current = self.alpha * previous - self.eta * self.calc_gradient(
                    x_data[row_index:row_index + 1],
                    y_data[row_index:row_index + 1]
                )
            else:
                raise Exception('unknown gradient type')
            previous = current
            self.w += previous
            self.update_loss_history(x_data, y_data)
            if abs(np.linalg.norm(current)) < self.tolerance:
                break
        return self

    def init_weights(self, x_data):
        if self.initial_weights is None:
            self.w = np.random.rand(x_data.shape[1])
        else:
            self.w = self.initial_weights

    def predict(self, x_data):
        if self.w is None:
            raise Exception('Not trained yet')
        else:
            return x_data.dot(self.w)

    def calc_gradient(self, x_data, y_data):
        distation = self.calc_distation(x_data, y_data)
        A = self.delta * x_data.T.dot((distation > self.delta).astype(float)) * (-1)
        B = self.delta * x_data.T.dot((distation < -self.delta).astype(float))
        C = np.dot(
            x_data.T, (-distation * (np.absolute(-distation) <= self.delta).astype(float)))
        return (A + B + C) / x_data.shape[0]

    def update_loss_history(self, x_data, y_data):
        self.loss_history.append(self.calc_loss(x_data, y_data))

    def calc_loss(self, x_data, y_data):
        distation = self.calc_distation(x_data, y_data)
        A = 0.5 * (distation) ** 2
        B = self.delta * np.absolute(distation) - 0.5 * self.delta ** 2
        return np.mean(np.where(np.absolute(distation) <= self.delta, A, B))

    def calc_distation(self, x_data, y_data):
        return y_data - x_data.dot(self.w)




