# -------------------------------------------------------------------------------
# Name: testing.py
# Purpose: This one has the function that I want to compare the civ to mil connection, all data is dummy data
#
# Author(s):    Megan Gordon
#
# Created:      06/04/2021
# Updated:      08/20/2021
#               03/15/2023
#
# Update Comment(s):
# manually putting in nodes with attributes (because I didn't know how to add the attributes)
# added more comments 8/20
# Added counters and percent change 3/15/23
#
# -------------------------------------------------------------------------------
from typing import Any, Union

import pandas as pd
from pandas import Series, DataFrame
from pandas.io.parsers import TextFileReader
from utils import *

import networkx as nx
import matplotlib
import numpy as np
import scipy
# import py2cytoscape
# import graphviz
import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# -----------------------------------------------

G = nx.Graph()      # create graph structure

# adding nodes with attributes M-military, C-civilian, TB-tech bridge
G.add_nodes_from([
    ("mil1", {"type": "M"}),
    ("mil3", {"type": "M"}),
    ("mil2", {"type": "M"}),
    ("civ1", {"type": "C"}),
    ("civ2", {"type": "C"}),
    ("TB", {"type": "TB"}),
])

# adding edges manually
G.add_edge("TB", "civ1")
G.add_edge("TB", "civ2")
G.add_edge("TB", "mil1")
G.add_edge("civ3", "civ1")
G.add_edge("mil1", "mil2")
G.add_edge("mil1", "civ1")
G.add_edge("mil3", "mil2")

print("All nodes with TB")
print(G.nodes)
print("All simple paths with TB")

# collect all nodes with attribute civ in A and all nodes with attribute mil in B
A = ["civ1", "civ2", "civ3"]
B = ["mil1", "mil2", "mil3"]

# pos = nx.random_layout(G["TB"])
pos = nx.circular_layout(G)
nx.draw_networkx_nodes(G, pos, nodelist=A, node_color="tab:blue")       # civilian nodes in blue
nx.draw_networkx_nodes(G, pos, nodelist=B, node_color="tab:orange")     # military nodes orange
nx.draw_networkx_nodes(G, pos, nodelist=["TB"], node_color="tab:red")   # techbridge nodes red
nx.draw_networkx_edges(G, pos, alpha=0.5, width=6)
#plt.show()

# !! THIS STEP !!!! vvv  is the important one!!!
#       It prints out the paths from each civilian in A to each military in B.
#       If using a directed graph, we would need to also count the military to civilian
#       Instead of printing the list, it would be better to have a counter instead, but I don't know how to do that
#       It may also be important to think about both how many total paths, as well as the existence of the path from
#           between node i to j  and see how these two variables change before and after TB


j = [1, 2, 3]
mil2civ = 0
num_paths = 0
for k in A:
    for h in B:
        paths = nx.all_simple_paths(G, source=k, target=h)
        length = len(list(paths))
        if length != 0:
            num_paths += 1
            mil2civ = mil2civ + length
print(num_paths)
print(mil2civ)


G.remove_nodes_from(["TB"])     # remove TB and all edges attached to TB
print("All nodes NO TB")
print(G.nodes)

# same as above but w/o TB
pos = nx.circular_layout(G)
nx.draw_networkx_nodes(G, pos, nodelist=["civ1", "civ2", "civ3"], node_color="tab:blue")
nx.draw_networkx_nodes(G, pos, nodelist=["mil1", "mil2", "mil3"], node_color="tab:orange")
nx.draw_networkx_edges(G, pos, alpha=0.5, width=6)
#plt.show()


print("All simple paths NO TB")
mil2civ_noTB = 0
num_paths_noTB = 0
for k in A:
    for h in B:
        paths = nx.all_simple_paths(G, source=k, target=h)
        length = len(list(paths))
        if length != 0:
            num_paths_noTB += 1
            mil2civ_noTB = mil2civ_noTB + length
print(num_paths_noTB)
print(mil2civ_noTB)

print("Percent Change: Paths exist between pairs of people:")
print((num_paths_noTB-num_paths)/num_paths)

print()
print("Percent Change: Total number of paths")
print((mil2civ_noTB-mil2civ)/mil2civ)
