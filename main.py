from datetime import datetime
import numpy as np
from keras.datasets import mnist
from MLP import *
from initializers import *
from optimizers import *
import matplotlib.pyplot as plt


def relu(x):
    return np.where(x > 0, x, 0)


def relu_p(x):
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_p(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_p(x):
    return 1 - tanh(x)**2


def softmax(X):
    nominator = np.exp(X)
    return nominator / np.sum(nominator, axis=1).reshape(len(nominator), 1)


def prepare_data(first_layer):
    (x_train_, y_train_), (x_test_, y_test_) = mnist.load_data()
    x_train_ = x_train_.reshape(x_train_.shape[0], first_layer) / 255
    x_test_ = x_test_.reshape(x_test_.shape[0], first_layer) / 255
    y_train_ = y_train_.reshape(y_train_.shape[0], 1)
    y_test_ = y_test_.reshape(y_test_.shape[0], 1)
    return x_train_, y_train_, x_test_, y_test_


# init variables
layers = (784, 100, 10)
functions_sigmoid = ((sigmoid, sigmoid_p), (softmax, None))
functions_relu = ((relu, relu_p), (relu, None))
X_train, y_train, X_test, y_test = prepare_data(first_layer=layers[0])

# init initializers
std_init = StandardInitializer(weights_mean=0, weights_sigma=0.1, bias_mean=0, bias_sigma=0.1, layers=layers)
xa = XavierInitializer(weights_mean=0, bias_mean=0, layers=layers)
he = HeInitializer(weights_mean=0, bias_mean=0, layers=layers)

# init optimizers
std_opt = StandardOptimizer()
momentum = MomentumOptimizer(gamma=0.7)
momentum_nest = MomentumNestOptimizer(gamma=0.7)
adagrad = AdagradOptimizer(epsilon=0.1)
adadelta = AdadeltaOptimizer(epsilon=0.0005, gamma=0.8)
adam = AdamOptimizer(epsilon=0.1, beta1=0.9, beta2=0.9)


show_tests = False
repeats = 10


# Experiments
def exp1():
    list_ = []
    optimizers = [
        StandardOptimizer(),
        MomentumOptimizer(gamma=0.9),
        MomentumNestOptimizer(gamma=0.9),
        AdagradOptimizer(epsilon=0.0001),
        AdadeltaOptimizer(epsilon=0.0005, gamma=0.9),
        AdamOptimizer(epsilon=0.1, beta1=0.9, beta2=0.9)
    ]
    for optimizer in optimizers:
        total = 0
        for i in range(repeats):
            # create model
            model = MultiLayerPerceptron(functions=functions_sigmoid, eta=0.1, stop_at=0.93,
                                         initializer=std_init,
                                         optimizer=optimizer)
            # fit model
            model.fit(X_train, y_train, X_test, y_test, package_size=100, max_eras=20)
            total += model.get_final_eras()
        list_.append(total / repeats)
    return ["Standard", "Momentum", "Momentum Nest.", "Adagrad", "Adadelta", "Adam"], list_


def exp2():
    list_ = []
    optimizers = [
        StandardOptimizer(),
        MomentumOptimizer(gamma=0.9),
        MomentumNestOptimizer(gamma=0.9),
        AdagradOptimizer(epsilon=0.0001),
        AdadeltaOptimizer(epsilon=0.0005, gamma=0.9),
        AdamOptimizer(epsilon=0.1, beta1=0.9, beta2=0.9)
    ]
    for optimizer in optimizers:
        total = 0
        for i in range(repeats):
            # create model
            model = MultiLayerPerceptron(functions=functions_relu, eta=0.1, stop_at=0.93,
                                         initializer=std_init,
                                         optimizer=optimizer)
            # fit model
            model.fit(X_train, y_train, X_test, y_test, package_size=100, max_eras=20)
            total += model.get_final_eras()
        list_.append(total / repeats)
    return ["Standard", "Momentum", "Momentum Nest.", "Adagrad", "Adadelta", "Adam"], list_


def exp3():
    list_ = []
    inits = [
        StandardInitializer(weights_mean=0, weights_sigma=0.1, bias_mean=0, bias_sigma=0.1, layers=layers),
        XavierInitializer(weights_mean=0, bias_mean=0, layers=layers),
        HeInitializer(weights_mean=0, bias_mean=0, layers=layers)
    ]
    for init in inits:
        total = 0
        for i in range(repeats):
            # create model
            model = MultiLayerPerceptron(functions=functions_sigmoid, eta=0.1, stop_at=0.93,
                                         initializer=init,
                                         optimizer=std_opt)
            # fit model
            model.fit(X_train, y_train, X_test, y_test, package_size=100, max_eras=20)
            total += model.get_final_eras()
        list_.append(total / repeats)
    return ["Standard", "Xavier", "He"], list_


def exp4():
    list_ = []
    inits = [
        StandardInitializer(weights_mean=0, weights_sigma=0.1, bias_mean=0, bias_sigma=0.1, layers=layers),
        XavierInitializer(weights_mean=0, bias_mean=0, layers=layers),
        HeInitializer(weights_mean=0, bias_mean=0, layers=layers)
    ]
    for init in inits:
        total = 0
        for i in range(repeats):
            # create model
            model = MultiLayerPerceptron(functions=functions_relu, eta=0.1, stop_at=0.93,
                                         initializer=init,
                                         optimizer=std_opt)
            # fit model
            model.fit(X_train, y_train, X_test, y_test, package_size=100, max_eras=20)
            total += model.get_final_eras()
        list_.append(total / repeats)
    return ["Standard", "Xavier", "He"], list_


def print_result(result_x, result_y, divider):
    print(f" {divider} ".join(map(str, result_x)))
    print(f" {divider} ".join(map(str, result_y)))


def make_exp(exp, xscale, title, x, y, dest, show, bar):
    print(f"Rozpoczeto {title.lower()}")
    exp_result = exp()
    print_result(*exp_result, "&")

    plt.bar(*exp_result) if bar else plt.plot(*exp_result)
    if xscale:
        plt.xscale(xscale)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show() if show else plt.savefig(dest)
    plt.clf()
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"Zakonczono {title.lower()}\n{dt_string}")


dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print(f"Start {dt_string}")
# make_exp(exp=exp1, xscale=None, title="Wpływ optymalizatorów na szybkość uczenia", x="", y="epoki", dest="plots/exp1.png", show=show_tests, bar=True)
# make_exp(exp=exp2, xscale=None, title="Wpływ optymalizatorów na szybkość uczenia", x="", y="epoki", dest="plots/exp2.png", show=show_tests, bar=True)
# make_exp(exp=exp3, xscale=None, title="Wpływ inicjalizatorów na szybkość uczenia", x="", y="epoki", dest="plots/exp3.png", show=show_tests, bar=True)
# make_exp(exp=exp4, xscale=None, title="Wpływ inicjalizatorów na szybkość uczenia", x="", y="epoki", dest="plots/exp4.png", show=show_tests, bar=True)
