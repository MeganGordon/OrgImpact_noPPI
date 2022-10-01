### NEED TO DO? DiD i UPDATE MY COMMENTS? ## 
# calculate centralities and average centralities (copy over code)
# save the average __centrality for each graph in a series object
# plot a histogram of that __centrality
# add random attributes for company type and then add civilian military dod attributes to them.
# military civilian mixing for each company, save in a series, plot histogram

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

def gen_array(n, k, e):
    num_graphs_generated = 1000
    graphsArray = []
    for i in range(num_graphs_generated):
        graph = generateG(n, k, e)
        graph = mutually_connect_tb(assign_random_attributes(graph))
        graphsArray.append(graph)
        graph_analysis(graph, i)
    return graphsArray

def generateG(n, k, e):  # n=#nodes k=#survey e=#edges.
    G = nx.DiGraph()
    # for i in range(num_node):
    #     G.add_node(i)
    for i in range(num_TB):
        G.add_node(i, type_of_node="TB")
    for i in range(num_surv-num_TB):
        G.add_node(i+num_TB, type_of_node="survey")
    for i in range(num_node-num_surv):
        G.add_node(i+num_surv, type_of_node="alter")
    outer = np.arange(num_surv, num_node, 1)
    flag = 0
    rng1 = np.random.default_rng()
    # "outer members" are alters who were not in the roster.
    for i in outer:  # instantiates the k+1 to n-1 outer nodes
        # print(i)
        l = rng1.integers(0, num_surv-1)
        G.add_edge(l, i)
        flag += 1
        # print(str(l) + ", " + str(i))
        # print(str(l), str(i))
    # now we need to consider alters who were in the roster (including TB members)
    extra = edg - flag
    # should this be k or k-1?
    for j in range(extra):
        rand_rost = rng1.integers(0, num_surv-1)  # from anyone in survey
        rand_node = rng1.integers(0, n - 1)  # to anyone at all #should this be only to TB and roster members
        while G.has_edge(rand_rost, rand_node):
            rand_rost = rng1.integers(0, num_surv - 1)
            rand_node = rng1.integers(0, n - 1)
        G.add_edge(rand_rost, rand_node)
        flag += 1
    return G


def assign_random_attributes(G):
    marker = 1
    integer_list = []
    rng1 = np.random.default_rng()
    random_num = rng1.integers(0, num_surv-1) #SCTB all took survey
    # G.add_node(str(random_num), military_status="Uniformed Service Member", type_of_node="TB")
    for i in range(num_node):
        integer_list.append(i)
    integer_list.remove(0)
    integer_list.remove(1)
    integer_list.remove(2)
    integer_list.remove(3)
    random.shuffle(integer_list)
    uniformed_members_not_tb = total_num_mil_incl_TB - num_mil_within_TB
    dod_not_tb = total_num_dod_incl_TB - num_DOD_within_TB
    j = 0
    for i in range(num_mil_within_TB):
        G.add_node(0, military_status="Uniformed Service Member")
        # don't add 1 to j counter because I removed the TB nodes
    for i in range(num_TB-1):
        G.add_node(i+1, military_status="Department of Defense (DoD)")
        # don't add 1 to j counter because I removed the TB nodes
    for i in range(uniformed_members_not_tb):
        G.add_node(integer_list[j], military_status="Uniformed Service Member")
        j += 1
    for i in range(total_num_civilians):
        G.add_node(integer_list[j], military_status="Civilian")
        j += 1
    for i in range(dod_not_tb):
        G.add_node(integer_list[j], military_status="Department of Defense (DoD)")
        j += 1
    return G

def mutually_connect_tb(G):
    # here, range(num_tb) works because the indexed nodes are assigned as TB
    for i in range(num_TB):
        for j in range(num_TB):
            if i != j:
                G.add_edge(i, j)
    return G

# returns df_new - indexed by names, with {betweenness, clustering, degree, eigenvector, katz}
def graph_analysis(gr, inte):
    global df_new
    connected_components_list.insert(inte, nx.number_connected_components(gr.to_undirected()))
    average_clustering_list.insert(inte, nx.average_clustering(gr))
    degree = nx.degree_centrality(gr)
    betweenness = nx.betweenness_centrality(gr)
    clustering = nx.clustering(gr)
    eigenvector = 0
    # eigenvector = nx.eigenvector_centrality_numpy(gr)
    katz = nx.katz_centrality(gr)
    data = {'betweenness': betweenness, 'clustering': clustering, 'degree': degree, 'eigenvector': eigenvector, 'katz': katz}
    # data = {'betweenness': betweenness, 'clustering': clustering, 'degree': degree, 'katz': katz}
    df_new = pd.DataFrame.from_dict(data)
    return df_new


def avg_analysis(G, G_noTB, integ):
    analyses_with_TB: pd.DataFrame() = graph_analysis(G, integ)
    mean_betweenness_withTB = 0
    mean_degree_withTB = 0
    mean_eigenvector_withTB = 0
    mean_katz_withTB = 0
    avg_clustering_withTB = 0
    civ_and_dod_to_mil_withTB = 0
    civ_to_dodandmil_withTB = 0
    for i in analyses_with_TB.index:
        mean_betweenness_withTB += analyses_with_TB.at[i, 'betweenness']
        mean_degree_withTB += analyses_with_TB.at[i, 'degree']
        mean_eigenvector_withTB += analyses_with_TB.at[i, 'eigenvector']
        mean_katz_withTB += analyses_with_TB.at[i, 'katz']
        avg_clustering_withTB += analyses_with_TB.at[i, 'clustering']
    avg_betweenness_list.insert(integ, mean_betweenness_withTB)
    avg_degree_list.insert(integ, mean_degree_withTB)
    avg_katz_list.insert(integ, mean_katz_withTB)
    analyses_no_TB: pd.DataFrame() = graph_analysis(G_noTB, integ)
    mean_betweenness_noTB = 0
    mean_degree_noTB = 0
    mean_eigenvector_noTB = 0
    mean_katz_noTB = 0
    avg_clustering_noTB = 0
    civ_and_dod_to_mil_noTB = 0
    civ_to_dodandmil_noTB = 0
    for i in analyses_no_TB.index:
        mean_betweenness_noTB += analyses_no_TB.at[i, 'betweenness']
        mean_degree_noTB += analyses_with_TB.at[i, 'degree']
        mean_eigenvector_noTB += analyses_with_TB.at[i, 'eigenvector']
        mean_katz_noTB += analyses_with_TB.at[i, 'katz']
        avg_clustering_noTB += analyses_with_TB.at[i, 'clustering']
    avg_betweenness_diff_list.insert(integ, abs(mean_betweenness_noTB - mean_betweenness_withTB) * 200 / (mean_betweenness_noTB + mean_betweenness_withTB))
    avg_degree_diff_list.insert(integ, abs(mean_degree_noTB - mean_degree_withTB) * 200 / (mean_degree_noTB + mean_degree_withTB))
    # avg_eigenvector_diff_list.insert(integ, abs(mean_eigenvector_noTB - mean_eigenvector_withTB) * 200 / (mean_eigenvector_noTB + mean_eigenvector_withTB))
    avg_katz_diff_list.insert(integ, abs(mean_katz_noTB - mean_katz_withTB) * 200 / (mean_katz_noTB + mean_katz_withTB))
    #avg_clustering_diff_list.insert(integ, abs(mean_betweenness_noTB - mean_betweenness_withTB) * 200 / (mean_betweenness_noTB + mean_betweenness_withTB))
    #civ_and_dod_to_mil_diff_list.insert(integ, abs(mean_betweenness_noTB - mean_betweenness_withTB) * 200 / (mean_betweenness_noTB + mean_betweenness_withTB))
    #civ_to_dod_and_mil_diff_list.insert(integ, abs(mean_betweenness_noTB - mean_betweenness_withTB) * 200 / (mean_betweenness_noTB + mean_betweenness_withTB))
    return

def percent_difference(a, b):
    percent_diff = abs(a - b) * 200 / (a + b)
    return percent_diff

def military_civilian_mixing(G, inte):
    mapping = {'Civilian': 0, 'Uniformed Service Member': 1, 'Department of Defense (DoD)': 2}
    mix_mat = nx.attribute_mixing_matrix(G, 'military_status', mapping=mapping)
    # print('Mixing value civilian to military (ignore dod): ')
    # print(mix_mat[mapping['Civilian'], mapping['Uniformed Service Member']] + mix_mat[mapping['Uniformed Service Member'], mapping['Civilian']])
    # print(mix_mat)
    ## identify together military (usm) and dod
    mapping = {'Civilian': 0, 'Uniformed Service Member': 1, 'Department of Defense (DoD)': 1}
    mix_mat = nx.attribute_mixing_matrix(G, 'military_status', mapping=mapping)
    # print('Mixing value civilian to (military and DOD): ')
    civ_to_dod_and_mil_mixing_list.insert(inte, mix_mat[mapping['Civilian'], mapping['Uniformed Service Member']] + mix_mat[mapping['Uniformed Service Member'], mapping['Civilian']])
    # print(mix_mat)
    ## idenitify together civilian and DOD categories
    mapping = {'Civilian': 0, 'Uniformed Service Member': 1, 'Department of Defense (DoD)': 0}
    mix_mat = nx.attribute_mixing_matrix(G, 'military_status', mapping=mapping)
    # print('Mixing value (civ and DOD) to military: ')
    civ_and_dod_to_mil_mixing_list.insert(inte, mix_mat[mapping['Civilian'], mapping['Uniformed Service Member']] + mix_mat[mapping['Uniformed Service Member'], mapping['Civilian']])
    # print(mix_mat)
    G.remove_node(0)
    G.remove_node(1)
    G.remove_node(2)
    G.remove_node(3)
    # now do the percent difference with TB removed and previous values
    mapping = {'Civilian': 0, 'Uniformed Service Member': 1, 'Department of Defense (DoD)': 1}
    mix_mat = nx.attribute_mixing_matrix(G, 'military_status', mapping=mapping)
    civ_to_dod_and_mil_diff_list.insert(inte, percent_difference(civ_to_dod_and_mil_mixing_list[inte], mix_mat[mapping['Civilian'], mapping['Uniformed Service Member']] + mix_mat[mapping['Uniformed Service Member'], mapping['Civilian']]))
    mapping = {'Civilian': 0, 'Uniformed Service Member': 1, 'Department of Defense (DoD)': 0}
    mix_mat = nx.attribute_mixing_matrix(G, 'military_status', mapping=mapping)
    civ_and_dod_to_mil_diff_list.insert(inte, percent_difference(civ_and_dod_to_mil_mixing_list[inte], mix_mat[mapping['Civilian'], mapping['Uniformed Service Member']] + mix_mat[mapping['Uniformed Service Member'], mapping['Civilian']]))

def plot_betweenness():
    plt.clf()
    plt.hist(avg_betweenness_list, bins='auto')
    plt.title("Distribution of Betweenness Centrality on Random Graphs with Constraints")
    plt.xlabel("Average Betweenness Centrality")
    plt.ylabel("Number of Occurences")
    plt.vlines(0.013794663048394393, 0, 150, colors='k', label='SCTB Avg Betweenness')
    plt.show()

    plt.clf()
    plt.hist(avg_betweenness_diff_list, bins='auto')
    plt.title("Distribution of Betweenness Centrality Percent Difference on Random Graphs with Constraints")
    plt.xlabel("Average Betweenness Centrality Percent Difference")
    plt.ylabel("Number of Occurences")
    plt.vlines(172.35753299420992, 0, 150, colors='k', label='SCTB Avg Betweenness')
    plt.show()
    return


def plot_katz():
    plt.clf()
    plt.hist(avg_katz_list, bins='auto')
    plt.title("Distribution of Katz Centrality on Random Graphs with Constraints")
    plt.xlabel("Average Katz Centrality")
    plt.ylabel("Number of Occurences")
    plt.vlines(8.132379733048001, 0, 150, colors='k', label='SCTB Avg Katz')
    plt.show()

    plt.clf()
    plt.hist(avg_katz_diff_list, bins='auto')
    plt.title("Distribution of Katz Centrality Percent Difference on Random Graphs with Constraints")
    plt.xlabel("Average Katz Centrality Percent Difference")
    plt.ylabel("Number of Occurrences")
    plt.vlines(28.571428571428353, 0, 150, colors='k', label='SCTB Avg Katz')
    plt.show()
    return


def plot_degree():
    plt.clf()
    plt.hist(avg_degree_list, bins=40)
    plt.title("Distribution of Degree Centrality on Random Graphs with Constraints")
    plt.xlabel("Average Degree Centrality")
    plt.ylabel("Number of Occurrences")
    plt.vlines(2.3283582089552213, 0, 150, colors='k', label='SCTB Avg Degree')
    plt.show()

    plt.clf()
    plt.hist(avg_degree_diff_list, bins='auto')
    plt.title("Distribution of Degree Centrality Percent Difference on Random Graphs with Constraints")
    plt.xlabel("Average Degree Centrality Percent Difference")
    plt.ylabel("Number of Occurences")
    plt.vlines(28.571428571428353, 0, 150, colors='k', label='SCTB Avg Degree')
    plt.show()
    return


def plot_eigenvector():
    plt.clf()
    plt.hist(avg_eigenvector_list, bins='auto')
    plt.title("Distribution of Eigenvector Centrality on Random Graphs with Constraints")
    plt.xlabel("Average Eigenvector Centrality")
    plt.ylabel("Number of Occurences")
    # plt.vlines(2.5298221281347035, 0, 150, colors='k', label='SCTB Avg EV')
    plt.show()

    plt.clf()
    plt.hist(avg_eigenvector_diff_list, bins='auto')
    plt.title("Distribution of Eigenvector Centrality Percent Difference on Random Graphs with Constraints")
    plt.xlabel("Average Eigenvector Centrality Percent Difference")
    plt.ylabel("Number of Occurences")
    # plt.vlines(119.99999999999996, 0, 150, colors='k', label='SCTB Avg EV')
    plt.show()
    return


def plot_mixing():
    plt.clf()
    plt.hist(civ_and_dod_to_mil_mixing_list, bins='auto')
    plt.title("Distr. of (Civilian and DOD) to (Military) Mixing on Random Graphs with Constraints")
    plt.xlabel("Mixing Value")
    plt.ylabel("Number of Occurences")
    plt.vlines(0.4571428571428572, 0, 150, colors='k', label='SCTB Avg EV')
    plt.show()

    plt.clf()
    plt.hist(civ_and_dod_to_mil_diff_list, bins='auto')
    plt.title("(Civilian and DOD) to (Military) Mixing Percent Difference on Random Graphs with Constraints")
    plt.xlabel("Percent Difference")
    plt.ylabel("Number of Occurences")
    # plt.vlines(119.99999999999996, 0, 150, colors='k', label='SCTB Avg EV')
    plt.show()

    plt.clf()
    plt.hist(civ_to_dod_and_mil_mixing_list, bins='auto')
    plt.title("Distr. of (Civilian) to (Military and DOD) Mixing on Random Graphs with Constraints")
    plt.xlabel("Mixing Value")
    plt.ylabel("Number of Occurences")
    plt.vlines(0.08974358974358974, 0, 150, colors='k', label='SCTB Avg EV')
    plt.show()

    plt.clf()
    plt.hist(civ_to_dod_and_mil_diff_list, bins='auto')
    plt.title("(Civilian) to (Military and DOD) Mixing Percent Difference on Random Graphs with Constraints")
    plt.xlabel("Percent Difference")
    plt.ylabel("Number of Occurences")
    # plt.vlines(119.99999999999996, 0, 150, colors='k', label='SCTB Avg EV')
    plt.show()
    return

def plot_histogram():
    plt.hist(connected_components_list, bins='auto')
    plt.title("Distribution of Connected Components on Random Graphs")
    plt.xlabel("Connected Components")
    plt.ylabel("Number of Occurences")
    plt.vlines(8, 0, 375, colors='k', label='SCTB Connected Components')
    # plt.show()

    plt.clf()
    plt.hist(average_clustering_list, bins='auto')
    plt.title("Distribution of Average Clustering on Random Graphs")
    plt.xlabel("Average Clustering Value")
    plt.ylabel("Number of Occurences")
    plt.vlines(0.01273342670401494, 0, 150, colors='k', label='SCTB Average Clustering')
    # plt.show()

    plot_degree()
    plot_betweenness()
    plot_katz()
    #plot_eigenvector()
    return


####################################################################
"""








"""
######################################################################################3

## SCTB ##
num_node = 68#number of nodes
num_surv = 17 #number of survey takersnet
edg = 68 #number of edges #calculated before edges were added between TB MEMBERES
#number of TB members

num_TB = 4
num_DOD_within_TB = 3
num_mil_within_TB = num_TB - num_DOD_within_TB

total_num_mil_incl_TB = 14
total_num_civilians = 34
total_num_dod_incl_TB = 20

average_clustering_list: list = []
connected_components_list: list = []
avg_betweenness_list: list = []
avg_katz_list: list = []
avg_degree_list: list = []
avg_eigenvector_list: list = []
civ_to_dod_and_mil_mixing_list: list = []
civ_and_dod_to_mil_mixing_list: list = []

#percent difference = |before removing TB - after removing TB| / .5*(before removing TB +after removing TB)
average_clustering_diff_list: list = []
connected_components_diff_list: list = []
avg_betweenness_diff_list: list = []
avg_katz_diff_list: list = []
avg_degree_diff_list: list = []
avg_eigenvector_diff_list: list = []
civ_to_dod_and_mil_diff_list: list = []
civ_and_dod_to_mil_diff_list: list = []

num_graphs_generated = 1000
graphsArray = []

for i in range(num_graphs_generated):
    graph = generateG(num_node, num_surv, edg)
    graph = mutually_connect_tb(assign_random_attributes(graph))
    graphsArray.append(graph)
    graph_analysis(graph, i)

graph_no_TB = graphsArray.copy()

for i in range(1000):
    graph_no_TB[i] = graphsArray[i].copy()
    graph_no_TB[i].remove_node(0)
    graph_no_TB[i].remove_node(1)
    graph_no_TB[i].remove_node(2)
    graph_no_TB[i].remove_node(3)
    avg_analysis(graphsArray[i], graph_no_TB[i], i)

#plot_histogram()

for i in range(1000):
    military_civilian_mixing(graphsArray[i], i)

plot_mixing()
"""
#np.array(X).tofile("C:/Users/megan/PycharmProjects/pythonProject3/File2.csv")

pos = nx.kamada_kawai_layout(graphsArray[0])
nx.draw_networkx_nodes(graphArray[0], pos, node_size=50, edgecolors="black")
nx.draw_networkx_edges(graphsArray[0], pos, alpha=0.5)
plt.show()
#nx.draw(hh.subgraph('TB'), pos=pos, node_color="red", edgecolors="black", node_size=100)
#nx.draw(hh.subgraph(roster), pos=pos, node_color="purple", edgecolors="black", node_size=50)

"""
