# -*- coding: utf-8 -*-
"""
Created on 21:59 2022/9/17

@author: shengrihui
"""

import matplotlib.pyplot as plt
import networkx as nx


class Node:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


node_x = Node('x')
node_k1 = Node('k1')
node_k2 = Node('k2')
node_b1 = Node('b1')
node_b2 = Node('b2')
node_linear1 = Node('linear1')
node_linear2 = Node('linear2')
node_sigmoid = Node('sigmoid')
node_y_true = Node('y_true')
node_loss = Node('loss')

computing_graph = {
    node_x: [node_linear1],
    node_k1: [node_linear1],
    node_b1: [node_linear1],
    node_linear1: [node_sigmoid],
    node_sigmoid: [node_linear2],
    node_k2: [node_linear2],
    node_b2: [node_linear2],
    node_linear2: [node_loss],
    node_y_true: [node_loss]
}

nx.draw(nx.DiGraph(computing_graph), with_labels=True)
plt.show()
