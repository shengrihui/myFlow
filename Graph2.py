# -*- coding: utf-8 -*-
"""
Created on 9:29 2022/9/18

@author: shengrihui
"""
import random
from functools import reduce
from operator import add

import matplotlib.pyplot as plt
import networkx as nx


class Node:
    def __init__(self, inputs=[], name=None):
        self.name = name
        self.inputs = inputs
        self.value = None
        self.outputs = []

        for node in self.inputs:
            node.outputs.append(self)

    def forward(self):
        print("get value from self")

    def backward(self):
        pass

    def __repr__(self):
        return self.name


class Placeholder(Node):
    def __init__(self, name):
        Node.__init__(self, [], name)

    def forward(self):
        print("Get value by human baba")

    def backward(self):
        print("Get gradient of loss to myself")


class Operator(Node):
    def __init__(self, name=None, inputs=[]):
        Node.__init__(self, inputs, name)

    def forward(self):
        print("Get value by {} and self function".format(' ,'.join(map(str, self.inputs))))

    def backward(self):
        print("Get gradient of # loss # to # {}".format(' ,'.join(map(str, self.inputs))))
        if self.outputs:
            for n in self.inputs:
                print(f"===>  ∂loss/∂{n} = ∂loss/∂{self} * ∂{self}/∂{n}")
                # 链式法则 ∂loss/∂{输入节点} = ∂loss/∂{该节点} * ∂{该节点}/∂{输入节点}


def topology(graph):
    sorted_node = []
    while graph:
        all_node_have_outputs = set(graph.keys())
        all_node_have_inputs = set(reduce(add, graph.values()))

        # node_has_no_input
        node_only_has_output = all_node_have_outputs - all_node_have_inputs

        if node_only_has_output:
            node = random.choice(list(node_only_has_output))
            sorted_node.append(node)
            if len(graph) == 1:
                sorted_node += graph[node]
            graph.pop(node)

            for _, links in graph.items():
                if node in links:
                    links.remove(node)
        else:
            raise TypeError("The graph cannot get topological oreder it has a cycle")

    return sorted_node


def forward_backward(order):
    for node in order:
        print(f"I am {node}")
        node.forward()
    print("*******************************")
    for node in order[::-1]:
        print(f"I am {node}")
        node.backward()


if __name__ == '__main__':
    node_x = Placeholder('x')
    node_k1 = Placeholder('k1')
    node_k2 = Placeholder('k2')
    node_b1 = Placeholder('b1')
    node_b2 = Placeholder('b2')
    node_linear1 = Operator('linear1', inputs=[node_x, node_k1, node_b1])
    node_sigmoid = Operator('sigmoid', inputs=[node_linear1])
    node_linear2 = Operator('linear2', inputs=[node_sigmoid, node_k2, node_b2])
    node_y_true = Placeholder('y_true')
    node_loss = Operator('loss', inputs=[node_y_true, node_linear2])

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

    # 自己观察得到顺序
    a_valid_order = [node_x, node_k1, node_b1, node_linear1, node_sigmoid,
                     node_k2, node_b2, node_linear2, node_y_true, node_loss]
    # print(a_valid_order)
    # forward_backward(a_valid_order)

    order = topology(computing_graph)
    print(order)
    forward_backward(order)
