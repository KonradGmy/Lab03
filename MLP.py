import numpy as np
from sklearn.utils import shuffle
from initializers import Initializer
from optimizers import StandardOptimizer


def y_to_vector(y, shape):
    result = np.zeros(shape)
    for i in range(result.shape[0]):
        result[i][y[i][0] - 1] = 1
    return result.T


class MultiLayerPerceptron:
    def __init__(self, eta, functions, stop_at, initializer: Initializer, optimizer: StandardOptimizer, report=False):
        self.A = []
        self.Z = []
        self.package_size = None
        self.final_eras = None
        self.stop_at = stop_at
        self.eta = eta
        self.functions = functions
        self.W, self.biases = initializer.initialize()
        self.optimizer = optimizer
        self.optimizer.initialize(self.W, self.biases)
        self.report = report

    def forward(self, X):
        a = X
        self.__init_forward(a)
        for i in range(len(list(zip(self.W, self.functions, self.biases)))):
            a = self.__calculate_and_save_activation(a, i)

        return self.A[-1]

    def backward(self, y_pred, y):
        self.optimizer.before_backward()
        deltas = self.__calculate_deltas(y, y_pred)
        deltas.reverse()
        self.__update_weights_and_biases(deltas)
        self.optimizer.after_backward()

    def fit(self, X, y, X_val, y_val, package_size, max_eras):
        biggest_mean_val, counter = self.__init_fit(package_size)

        for e in range(max_eras):
            X, X_val, fit, fit_val, y, y_val = self.__init_era_and_shuffle(X, X_val, y, y_val)
            self.__fit_for_test(X, fit, package_size, y)
            self.__fit_for_valid(X_val, fit_val, package_size, y_val)

            counter, mean, mean_val, biggest_mean_val = \
                self.__calc_means_and_manage_stop(biggest_mean_val, counter, fit, fit_val)
            if counter > 5 or (self.stop_at and mean_val > self.stop_at):
                self.final_eras = e
                break

            if self.report:
                print(f"Era {e + 1}  accuracy: {round(mean, 4)} val accuracy: {round(mean_val, 4)}")

    # forward

    def __init_forward(self, a):
        self.A.clear()
        self.Z.clear()
        self.A.append(a.T)

    def __calculate_and_save_activation(self, a, i):
        act_function, bias, w = self.__use_optimizer(i)
        a, z = self.__calculate_activation(a, act_function, bias, w)
        self.__save_activation(a, z)
        return a

    def __use_optimizer(self, i):
        w_add, bias_add = self.optimizer.get_W_and_bias_for_forward(i)
        w = self.W[i] + w_add
        bias = self.biases[i] + bias_add
        act_function = self.functions[i]
        return act_function, bias, w

    def __calculate_activation(self, a, act_function, bias, w):
        z = a.dot(w) + bias
        a = act_function[0](z)
        return a, z

    def __save_activation(self, a, z):
        self.Z.append(z.T)
        self.A.append(a.T)

    # backward

    def backward(self, y_pred, y):
        self.optimizer.before_backward()
        deltas = self.__calculate_deltas(y, y_pred)
        deltas.reverse()
        self.__update_weights_and_biases(deltas)
        self.optimizer.after_backward()

    def __calculate_deltas(self, y, y_pred):
        delta = y - y_pred
        deltas = [delta]
        for i in range(len(self.W) - 2, -1, -1):
            deltas.append(self.W[i + 1].dot(deltas[-1]) * self.functions[i][1](self.Z[i]))
        return deltas

    def __update_weights_and_biases(self, deltas):
        for i in range(len(self.W)):
            w_delta = self.optimizer.optimaze_W(i, self.eta / self.package_size * deltas[i].dot(self.A[i].T).T)
            bias_delta = self.optimizer.optimaze_bias(i, self.eta / self.package_size * deltas[i].sum(axis=1))
            self.W[i] += w_delta
            self.biases[i] += bias_delta

    # fit

    def __calc_means_and_manage_stop(self, biggest_mean_val, counter, fit, fit_val):
        mean = np.array(fit).mean()
        mean_val = np.array(fit_val).mean()
        if mean_val < biggest_mean_val:
            counter += 1
        else:
            biggest_mean_val = mean_val
            counter = 0
        return counter, mean, mean_val, biggest_mean_val

    def __init_fit(self, package_size):
        self.package_size = package_size
        self.final_eras = 0
        self.optimizer.on_fit_start()
        biggest_mean_val = 0
        counter = 0
        return biggest_mean_val, counter

    def __init_era_and_shuffle(self, X, X_val, y, y_val):
        fit = []
        fit_val = []
        X, y = shuffle(X, y, random_state=0)
        X_val, y_val = shuffle(X_val, y_val, random_state=0)
        return X, X_val, fit, fit_val, y, y_val

    def __cut_package(self, X, i, package_size, y):
        x = X[i * package_size:(i + 1) * package_size]
        y_ = y_to_vector(y[i * package_size:(i + 1) * package_size], (package_size, self.W[-1].shape[1]))
        return x, y_

    def __fit_for_test(self, X, fit, package_size, y):
        for i in range(int((len(X)) / package_size) - 1):
            x, y_ = self.__cut_package(X, i, package_size, y)
            forward_result = self.forward(x)
            self.backward(forward_result, y_)
            fit.append(np.where(forward_result.argmax(axis=0) == y_.argmax(axis=0), 1, 0).mean())

    def __fit_for_valid(self, X_val, fit_val, package_size, y_val):
        for i in range(int((len(X_val)) / package_size) - 1):
            x, y_ = self.__cut_package(X_val, i, package_size, y_val)
            forward_result = self.forward(x)
            fit_val.append(np.where(forward_result.argmax(axis=0) == y_.argmax(axis=0), 1, 0).mean())

    def get_final_eras(self):
        return self.final_eras


