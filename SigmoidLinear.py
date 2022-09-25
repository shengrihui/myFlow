# -*- coding: utf-8 -*-
"""
Created on 16:48 2022/9/17

@author: shengrihui
"""

import random

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


sub_x = np.linspace(-10, 10)


# print(sub_x.shape)
# print(sub_x)


# plt.plot(sub_x, sigmoid(sub_x))
# plt.title("sigmoid")
# plt.show()


def random_linear(x):
    k, b = random.normalvariate(0, 1), random.normalvariate(0, 1)
    return k * x + b


# for _ in range(10):
#     plt.plot(sub_x, random_linear(sub_x))
# plt.title("linear")
# plt.show()
#
# for _ in range(10):
#     plt.plot(sub_x, sigmoid(random_linear(sub_x)))
# plt.title("sigmoid-linear")
# plt.show()
#
# for _ in range(10):
#     plt.plot(sub_x, random_linear(sigmoid(random_linear(sub_x))))
# plt.title("linear-sigmoid-linear")
# plt.show()

from matplotlib.animation import FuncAnimation


def get_random_function(xs, f):
    index = random.randint(0, len(xs))
    return np.concatenate([f(xs[:index]), f(xs[index:])])


def animate(i):
    fig.clear()
    # plt.plot(sub_x, get_random_function(sub_x, random_linear))
    plt.plot(sub_x, get_random_function(get_random_function(sub_x, random_linear),sigmoid))
    plt.plot(sub_x, random_linear(sigmoid(random_linear(sub_x))))


fig = plt.gcf()
ani = FuncAnimation(fig, animate, interval=400)
plt.show()
