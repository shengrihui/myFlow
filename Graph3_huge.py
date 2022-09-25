# -*- coding: utf-8 -*-
"""
Created on 15:05 2022/9/18

@author: shengrihui
"""
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import numpy as np
from Graph2 import topology


# 生成随机图
def generate_random_graph(node_num=100):
    nodes = [f"node_{i}" for i in range(node_num)]
    random_graph = defaultdict(list)
    random.shuffle(nodes)
    for i, n in enumerate(nodes):
        if i < len(nodes) - 1:
            random_graph[n] += random.sample(nodes[i + 1:],  # 输出节点保证为1到3个
                                             k=random.randint(1, min(len(nodes) - (i + 1), 3)))
    return random_graph


original_graph = generate_random_graph(node_num=30)
r_g = nx.DiGraph(original_graph)
r_g_layout = nx.layout.spring_layout(r_g)
# nx.draw(r_g, r_g_layout, node_size=20)
# plt.show()

topological_order_for_huge = topology(original_graph)

color = ('blue', 'red')
# color = ('red', 'blue')
before, changed = color
visited_order = topological_order_for_huge


def all_step():
    width = 3
    fig, ax = plt.subplots(len(visited_order) // width + 1, width, figsize=(30, 30))

    for i, node in enumerate(visited_order):
        ix = np.unravel_index(i, ax.shape)
        # numpy.unravel_index()函数的作用是获取一个/组int类型的索引值在一个多维数组中的位置。
        plt.sca(ax[ix])
        ax[ix].set_title("Feed forward step:{}".format(i))

        map_colors = [changed if node in visited_order[:i + 1] else before for node in r_g]
        nx.draw(r_g, r_g_layout, node_color=map_colors, node_size=20)
    plt.show()


def flash():
    def animate(step):
        fig.clear()
        map_colors = [changed if node in visited_order[:step] else before for node in r_g]
        if step < len(visited_order):
            ax.set_title('get node: ∂(loss)/∂{}.inputs value'.format(visited_order[step]))
        nx.draw(r_g, r_g_layout, node_color=map_colors, with_labels=True, font_size=5, node_size=150)

    ax = plt.gca()
    fig = plt.gcf()
    ani = FuncAnimation(fig, animate, interval=100)
    plt.show()


all_step()
# flash()
