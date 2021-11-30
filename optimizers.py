import numpy as np


class StandardOptimizer:
    def initialize(self, W, biases):
        pass

    def optimaze_W(self, i, gradient):
        return gradient

    def optimaze_bias(self, i, gradient):
        return gradient

    def on_fit_start(self):
        pass

    def before_backward(self):
        pass

    def after_backward(self):
        pass

    def get_W_and_bias_for_forward(self, i):
        return 0, 0


class AdagradOptimizer(StandardOptimizer):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.sum_of_prev_gradsW = None
        self.sum_of_prev_grads_bias = None

    def initialize(self, W, biases):
        self.sum_of_prev_gradsW = [np.zeros_like(w_i) for w_i in W]
        self.sum_of_prev_grads_bias = [np.zeros_like(b_i) for b_i in biases]

    def optimaze_W(self, i, gradient):
        self.sum_of_prev_gradsW[i] += np.square(gradient)
        multiplier = 1 / np.sqrt(self.sum_of_prev_gradsW[i] + self.epsilon)
        return multiplier * gradient

    def optimaze_bias(self, i, gradient):
        self.sum_of_prev_grads_bias[i] += np.square(gradient)
        multiplier = 1 / np.sqrt(self.sum_of_prev_grads_bias[i] + self.epsilon)
        return multiplier * gradient


class AdadeltaOptimizer(StandardOptimizer):
    def __init__(self, epsilon, gamma):
        self.epsilon = epsilon
        self.gamma = gamma

        self.expected_g_W = None
        self.expected_g_bias = None

        self.expected_w_W = None
        self.expected_w_bias = None

        self.sum_of_prev_gradsW = None
        self.sum_of_prev_grads_bias = None

    def initialize(self, W, biases):
        self.expected_g_W = [np.zeros_like(w_i) for w_i in W]
        self.expected_g_bias = [np.zeros_like(b_i) for b_i in biases]

        self.expected_w_W = [np.zeros_like(w_i) for w_i in W]
        self.expected_w_bias = [np.zeros_like(b_i) for b_i in biases]

        self.sum_of_prev_gradsW = [np.zeros_like(w_i) for w_i in W]
        self.sum_of_prev_grads_bias = [np.zeros_like(b_i) for b_i in biases]

    def optimaze_W(self, i, gradient):
        E_g = self.gamma * self.expected_g_W[i].mean() + (1 - self.gamma) * np.square(gradient)

        delta_w = gradient / np.sqrt(E_g.mean() + self.epsilon)
        E_w = self.gamma * self.expected_w_W[i].mean() + (1 - self.gamma) * np.square(delta_w)

        RMS_w = np.sqrt(E_w)
        RMS_g = np.sqrt(E_g + self.epsilon)
        output = RMS_w / RMS_g * gradient

        self.expected_g_W[i] = E_g
        self.expected_w_W[i] = E_w

        return output

    def optimaze_bias(self, i, gradient):
        E_g = self.gamma * self.expected_g_bias[i].mean() + (1 - self.gamma) * np.square(gradient)

        delta_w = gradient / np.sqrt(E_g.mean() + self.epsilon)
        E_w = self.gamma * self.expected_w_bias[i].mean() + (1 - self.gamma) * np.square(delta_w)

        RMS_w = np.sqrt(E_w)
        RMS_g = np.sqrt(E_g + self.epsilon)
        output = RMS_w / RMS_g * gradient

        self.expected_g_bias[i] = E_g
        self.expected_w_bias[i] = E_w

        return output


class AdamOptimizer(StandardOptimizer):
    def __init__(self, epsilon, beta1, beta2):
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 1

        self.m_W = None
        self.m_bias = None
        self.v_W = None
        self.v_bias = None

    def initialize(self, W, biases):
        self.m_W = [np.zeros_like(w_i) for w_i in W]
        self.m_bias = [np.zeros_like(b_i) for b_i in biases]
        self.v_W = [np.zeros_like(w_i) for w_i in W]
        self.v_bias = [np.zeros_like(b_i) for b_i in biases]

    def optimaze_W(self, i, gradient):
        self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * gradient
        self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (gradient ** 2)

        m_corr = self.m_W[i] / (1 - self.beta1 ** self.t)
        v_corr = self.v_W[i] / (1 - self.beta2 ** self.t)

        return m_corr / (np.sqrt(v_corr) + self.epsilon)

    def optimaze_bias(self, i, gradient):
        self.m_bias[i] = self.beta1 * self.m_bias[i] + (1 - self.beta1) * gradient
        self.v_bias[i] = self.beta2 * self.v_bias[i] + (1 - self.beta2) * (gradient ** 2)

        m_corr = self.m_bias[i] / (1 - self.beta1 ** self.t)
        v_corr = self.v_bias[i] / (1 - self.beta2 ** self.t)

        self.t += 1

        return m_corr / (np.sqrt(v_corr) + self.epsilon)


class MomentumOptimizer(StandardOptimizer):
    def __init__(self, gamma):
        self.gamma = gamma
        self.last_W_add = None
        self.last_bias_add = None
        self.return_last_W_add = None
        self.return_last_bias_add = None

    def optimaze_W(self, i, gradient):
        result = gradient
        if self.last_W_add:
            result += self.gamma * self.last_W_add[i]
        self.return_last_W_add.append(result)
        return result

    def optimaze_bias(self, i, gradient):
        result = gradient
        if self.last_W_add:
            result += self.gamma * self.last_bias_add[i]
        self.return_last_bias_add.append(result)
        return result

    def before_backward(self):
        self.return_last_W_add = []
        self.return_last_bias_add = []

    def after_backward(self):
        self.last_W_add = self.return_last_W_add
        self.last_bias_add = self.return_last_bias_add


class MomentumNestOptimizer(MomentumOptimizer):
    def get_W_and_bias_for_forward(self, i):
        if self.last_W_add:
            return self.last_W_add[i], self.last_bias_add[i]
        else:
            return 0, 0
