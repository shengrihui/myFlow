# -*- coding: utf-8 -*-
"""
Created on 15:31 2022/9/17

@author: shengrihui
"""
import random
from collections import defaultdict
from Graph2 import topology

import numpy as np


class Node:
    def __init__(self, inputs=[], name=None, is_trainable=False):
        self.inputs = inputs
        self.outputs = []

        for n in self.inputs:
            n.outputs.append(self)
            # 将该节点self加入到输入的节点的outputs里

        self.name = name
        self.value = None

        self.gradients = dict()
        # {输入的节点:该节点对输入节点的求导}
        self.is_trainable = is_trainable

    def forward(self):
        """
        Forward propagation.
        前向传播
        基于输入节点计算输出，并将结果存于 self.value
        """
        return NotImplemented

    def backward(self):
        raise NotImplemented


class Placeholder(Node):
    def __init__(self, name=None, is_trainable=False):
        Node.__init__(self, name=name, is_trainable=is_trainable)
        # self.name = name
        # self.is_trainable = is_trainable
        # super(Placeholder, self).__init__(name=name, is_trainable=is_trainable)

    def backward(self):
        self.gradients[self] = self.outputs[0].gradients[self]
        # print(f"∂loss/∂{self.name} ={self.gradients[self]}")

    def __repr__(self):
        return f"Placeholder {self.name}"


class Linear(Node):
    def __init__(self, x, k, b, name=None):
        Node.__init__(self, inputs=[x, k, b], name=name)
        # self.inputs = [x, k, b]
        # self.name = name
        # super(Linear, self).__init__(inputs=[x, k, b], name=name)

    def forward(self):
        x, k, b = self.inputs
        self.value = k.value * x.value + b.value

    def backward(self):
        x, k, b = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self] * k.value
        self.gradients[k] = self.outputs[0].gradients[self] * x.value
        self.gradients[b] = self.outputs[0].gradients[self] * 1

    def __repr__(self):
        return f"Linear  {self.name}"


class Sigmoid(Node):
    def __init__(self, x, name=None):
        Node.__init__(self, inputs=[x], name=name)
        # self.inputs = [x]
        # self.name = name
        # super(Sigmoid, self).__init__(inputs=[x], name=name)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        # print(self.inputs[0].value)
        self.value = self._sigmoid(self.inputs[0].value)
        # print(self.value)
        # print("******")

    def backward(self):
        x = self.inputs[0]
        # # sigmoid' = y(1-y)
        # self.gradients[x] = self.outputs[0].gradients[self] * self._sigmoid(x.value) * (1 - self._sigmoid(x.value))
        self.gradients[x] = self.outputs[0].gradients[self] * (np.exp(-x.value) / np.square(1 + np.exp(-x.value)))
        # print(f"sigmoid.value={self.value}")
        # print(f"∂loss/∂{x.name} ={self.gradients[x]}")

    def __repr__(self):
        return f"Sigmoid  {self.name}"


class Loss(Node):
    def __init__(self, y, y_hat, name=None):
        Node.__init__(self, inputs=[y, y_hat], name=name)
        # super(Loss, self).__init__(inputs=[y, y_hat], name=name)

    def forward(self):
        y = self.inputs[0]
        y_hat = self.inputs[1]
        self.value = np.mean(np.square(y.value - y_hat.value))

    def backward(self):
        y = self.inputs[0]
        y_hat = self.inputs[1]
        self.gradients[y] = 2 * np.mean(y.value - y_hat.value)
        self.gradients[y_hat] = -2 * np.mean(y.value - y_hat.value)
        # print(f"∂loss/∂{y.name} ={self.gradients[y]}")
        # print(f"∂loss/∂{y_hat.name} ={self.gradients[y_hat]}")

    def __repr__(self):
        return f"Loss {self.name}"


def convert_feed_dict_to_graph(feed_dict):
    """
    狗崽计算图
    """
    # 传入的feed_dict是边缘的叶子节点，
    # 也就是在构建图的过程的当中，有了输入还没输出、还需要向外延伸的节点
    need_expand = [n for n in feed_dict]
    computing_graph = defaultdict(list)

    while need_expand:
        n = need_expand.pop(0)  # 取出列表中的第一个，这样其他的节点可以append
        if n in computing_graph:
            continue
        if isinstance(n, Placeholder):  # 给Placeholder节点赋值
            n.value = feed_dict[n]
        # 将n的输出节点接到计算图后面，也加入到需要扩展的列表中
        # 因为不同的节点会有同一个的输出（比如x,k,b的linear），
        # 所以前面要判断从需要扩展的列表中取出n是否已经在计算图当中
        for m in n.outputs:  # 在创建对象的时，当前对象会将自己self加入到输入节点inputs的输出节点outputs上
            computing_graph[n].append(m)  # 接到计算图中n节点的下一个节点上
            need_expand.append(m)  # 那下一个节点加入到继续往外延伸的列表当中
    return computing_graph


def forward(graph_sorted_nodes):
    for node in graph_sorted_nodes:
        # print(f"I am {node.name}")
        node.forward()
        # if isinstance(node, Loss):
        #     print(f"loss value {node.value}")


def backward(graph_sorted_nodes):
    for node in graph_sorted_nodes[::-1]:
        # print(f"I am {node.name}")
        node.backward()


def run_one_epoch(graph_sorted_nodes):
    forward(graph_sorted_nodes)
    backward(graph_sorted_nodes)


def optimize(graph_sorted_nodes, learning_rate=1e-3):
    for node in graph_sorted_nodes:
        if node.is_trainable:
            v = node.value
            node.value -= learning_rate * node.gradients[node]
            # print(f"node {node.name} value before;{v} grad:{node.gradients[node]}  after: {node.value}")
            cmp = "large" if node.gradients[node] > 0 else "small"


if __name__ == '__main__':
    node_x = Placeholder(name="x")
    node_y = Placeholder(name="y")

    node_k = Placeholder(name="k", is_trainable=True)
    node_b = Placeholder(name="b", is_trainable=True)

    node_linear = Linear(x=node_x, k=node_k, b=node_b, name="linear")
    node_sigmoid = Sigmoid(x=node_linear, name="sigmoid")
    node_loss = Loss(y_hat=node_sigmoid, y=node_y, name="loss")

    # feed_dict = {
    #     node_x: 30,
    #     node_y: 10,
    #     node_k: 123,
    #     node_b: 110.38
    # }
    feed_dict = {
        node_x: 3,
        node_y: random.random(),
        node_k: random.random(),
        node_b: 0.38
    }

    # node_list=[node_x,node_y,node_k,node_b,node_loss,node_linear,node_sigmoid]
    # for n in node_list:
    #     print(n,n.outputs)

    nodes = convert_feed_dict_to_graph(feed_dict)
    # print(nodes)
    sorted_nodes = topology(nodes)
    # print(sorted_node)

    epochs = 1000
    loss_history = []
    for n in feed_dict:
        print(n, n.value)
    for epoch in range(epochs):
        run_one_epoch(sorted_nodes)
        _loss_node = sorted_nodes[-1]
        assert isinstance(_loss_node, Loss)
        loss_value = _loss_node.value
        loss_history.append(loss_value)
        if epoch % 100 == 0:
            print(f"epoch: {epoch} loss value: {loss_value}")
        optimize(sorted_nodes, learning_rate=1e-3)

    for n in feed_dict:
        print(n, n.value)

    # import tqdm
    # import time
    # for _ in tqdm.tqdm(range(60)):
    #     time.sleep(1)


