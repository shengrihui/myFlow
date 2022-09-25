# -*- coding: utf-8 -*-
"""
Created on 19:13 2022/9/24

@author: shengrihui
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 1000)
# print(x)
y = 1 / (1 + np.exp(-x))
g1 = np.exp(-x) / (np.exp(-x) + 1) ** 2
g2 = y * (1 - y)
plt.plot(x, g1)
# plt.plot(x, y)
plt.show()
plt.plot(x, g2)
plt.show()
