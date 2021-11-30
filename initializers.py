import numpy as np
from math import sqrt
from abc import abstractmethod


class Initializer:
    def initialize(self):
        return self._initialize_W(), self._initialize_bias()

    @abstractmethod
    def _initialize_W(self):
        pass

    @abstractmethod
    def _initialize_bias(self):
        pass


class StandardInitializer(Initializer):
    def __init__(self, weights_mean, weights_sigma, bias_mean, bias_sigma, layers):
        self.weights_mean = weights_mean
        self.weights_sigma = weights_sigma
        self.bias_mean = bias_mean
        self.bias_sigma = bias_sigma
        self.layers = layers

    def _initialize_W(self):
        return [np.random.normal(
            self.weights_mean,
            self.weights_sigma,
            size=(self.layers[i], self.layers[i + 1]))
            for i in range(len(self.layers) - 1)]

    def _initialize_bias(self):
        return [np.random.normal(
            self.bias_mean,
            self.bias_sigma,
            size=(1, self.layers[i + 1]))
            for i in range(len(self.layers) - 1)]


class XavierInitializer(Initializer):
    def __init__(self, weights_mean, bias_mean, layers):
        self.weights_mean = weights_mean
        self.bias_mean = bias_mean
        self.layers = layers

    def _initialize_W(self):
        return [np.random.normal(
            self.weights_mean,
            sqrt(2.0 / (self.layers[i+1] + self.layers[i])),
            size=(self.layers[i], self.layers[i + 1]))
            for i in range(len(self.layers) - 1)]

    def _initialize_bias(self):
        return [np.random.normal(
            self.bias_mean,
            sqrt(2.0 / (self.layers[i+1] + self.layers[i])),
            size=(1, self.layers[i + 1]))
            for i in range(len(self.layers) - 1)]


class HeInitializer(Initializer):
    def __init__(self, weights_mean, bias_mean, layers):
        self.weights_mean = weights_mean
        self.bias_mean = bias_mean
        self.layers = layers

    def _initialize_W(self):
        return [np.random.normal(
            self.weights_mean,
            sqrt(2.0 / (self.layers[i])),
            size=(self.layers[i], self.layers[i + 1]))
            for i in range(len(self.layers) - 1)]

    def _initialize_bias(self):
        return [np.random.normal(
            self.bias_mean,
            sqrt(2.0 / (self.layers[i])),
            size=(1, self.layers[i + 1]))
            for i in range(len(self.layers) - 1)]
