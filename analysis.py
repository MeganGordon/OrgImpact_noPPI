from typing import List
import numpy as np
import networkx as nx
from networkx import DiGraph
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
import pandas as pd


## FUNCTIONS ################################

def company_stuff(section, G):
    # takes in section2 and nx.DiGraph, sets the company name for the connections and type_of_node = pendant
    dictofjobs: dict = {}
    for j in section.keys():
        if section.loc[j].get('q1') != []:
            for i in range(len(section.loc[j].get('q1'))):
                dictofjobs[section.loc[j].get('q1')[i].get('name')] = section.loc[j].get('q1')[i].get('org')
        if section.loc[j].get('q2') != []:
            for k in range(len(section.loc[j].get('q2'))):
                dictofjobs[section.loc[j].get('q2')[k].get('name')] = section.loc[j].get('q2')[k].get('org')
            #    print()
    # ##### NEED TO CLEAN UP company names HERE, combine and  #####
    # sets the node attribute to 'pendant' if the people are not already 'TB' or 'roster'
    nx.set_node_attributes(G, dictofjobs, name='company')
    type_of_node_temp: dict = nx.get_node_attributes(G, "type_of_node")
    dictofpendants: dict = {}
    for i in G.nodes:
        dictofpendants[i] = 'pendant'
    for j in type_of_node_temp:
        if isinstance(type_of_node_temp[j], str):
            dictofpendants[j] = type_of_node_temp[j]
    nx.set_node_attributes(G, dictofpendants, name="type_of_node")
    return G


def adding_company_and_mil_status_to_roster(df):
    # creates a nx Digraph and adds edges from roster to each supervisor
    # should probably add attribute to edge for "boss"
    G = nx.DiGraph()
    for i in df.index:
        G.add_node(df.fullName[i], company=df.companyName[i])
        for k in df.organizationRole[i]:
            G.add_node(df2.fullName[i], company_type=k)
            if k == 'Uniformed Service Member':
                G.add_node(df.fullName[i], military_status='Uniformed Service Member')
            if k == 'Non-DoD Federal Government':
                G.add_node(df.fullName[i], military_status='Civilian')
            if k == 'Private Sector':
                G.add_node(df.fullName[i], military_status='Civilian')
            if k == 'Academia':
                G.add_node(df.fullName[i], military_status='Civilian')
            if k == 'State and Local Government':
                G.add_node(df.fullName[i], military_status='Civilian')
            if k == 'Nonprofit':
                G.add_node(df.fullName[i], military_status='Civilian')
            if k == 'Department of Defense (DoD)':
                G.add_node(df.fullName[i], military_status='Department of Defense (DoD)')
    return G


# REMOVED contract_TB()

def bar_graphs(df):
    ax = df.plot.bar()
    # plt.show()
    hist = df.hist(bins=30, log=True)
    plt.show()

# REMOVED attr_for_NWTB()
# REMOVED add_edges_between_NWTB()

def concat_name(df):
    fullName = pd.Series(len(df.index), name='fullName')

    for i in df.index:
        fullName[i] = df["firstName"][i] + " " + df["lastName"][i]
    df = df.merge(fullName, left_index=True, right_index=True)
    return df


def add_supervisors_pendants(df, G):
    supervisor_dict: dict = {}
    for i in df.index:
        A: List = df.supervisors[i].split(",")
        # print(A)
        for k in A:
            if k != "n/a":
                if k != "N/A":
                    G.add_node(k, company=df.companyName[i])
                    G.add_edge(df2.fullName[i], k)
                    supervisor_dict[k] = 'supervisor'
                    for l in df.organizationRole[i]:
                        G.add_node(k, company_type=l)
                        if l == 'Uniformed Service Member':
                            G.add_node(k, military_status='Uniformed Service Member')
                        if l == 'Non-DoD Federal Government':
                            G.add_node(k, military_status='Civilian')
                        if l == 'Private Sector':
                            G.add_node(k, military_status='Civilian')
                        if l == 'Academia':
                            G.add_node(k, military_status='Civilian')
                        if l == 'State and Local Government':
                            G.add_node(k, military_status='Civilian')
                        if l == 'Nonprofit':
                            G.add_node(k, military_status='Civilian')
                        if l == 'Department of Defense (DoD)':
                            G.add_node(k, military_status='Department of Defense (DoD)')
    G.remove_node('None')
    nx.set_node_attributes(G, supervisor_dict, "type_of_node")
    # ### ADD COMPANY OF SUPERVISORS BY COMPANY OF THESE PEEPS
    # type_of_node_list = []
    # for i in Graw.nodes:
    #     nx.set_node_attributes(Graw, type_of_node_list, "type_of_node")
    # # print(type(df2.section2[0]['q1']))
    # # print(df2.section2.index)
    roster = []
    for i in df.section2.index:
        list.append(roster, df.fullName[i])
        #    print(type(df.section2[i]))
        for j in df.section2[i]:
            for k in df.section2[i][j]:
                #            print(df.section2[i][j][k])
                G.add_node(k['name'], question=j)
                G.add_edge(df.fullName[i], k['name'], relationship=k['relationship'])
                G.add_node(df.fullName[i], type_of_node="roster")
                # could separate q1 and q2 here
    return G


## REMOVED add_unknown_company_type() 

def plot_one_graph(h):
    civilian = []
    uniformed = []
    dod = []
    military_status: dict = nx.get_node_attributes(h, 'military_status')
    for i in military_status:
        if military_status[i] == 'Civilian':
            civilian.append(i)
        if military_status[i] == 'Department of Defense (DoD)':
            dod.append(i)
        if military_status[i] == 'Uniformed Service Member':
            uniformed.append(i)
    pos = nx.kamada_kawai_layout(h)
    nx.draw_networkx_nodes(h, pos, node_size=50, edgecolors="black")
    nx.draw_networkx_edges(h, pos, alpha=0.4)
    # nx.draw_networkx_labels(h, pos)
    nx.draw(h.subgraph(civilian), pos=pos, node_color="red", edgecolors="black", node_size=100)
    nx.draw(h.subgraph(dod), pos=pos, node_color="blue", edgecolors="black", node_size=100)
    nx.draw(h.subgraph(uniformed), pos=pos, node_color="green", edgecolors="black", node_size=100)
    # nx.draw(h.subgraph(roster), pos=pos, node_color="purple", node_size=50, edgecolors="black")
    plt.show()

## REMOVED plot_combined() function 

# returns df - indexed by names, with {betweenness, clustering, degree, eigenvector, katz}
def graph_analysis(gr):
    global df_new
    names: List = gr.nodes
    if nx.connected_components(gr.to_undirected()) == 1:
        center = nx.center(gr.to_undirected())
        conn_comp = 0
    else:
        conn_comp = 0
        for i in nx.connected_components(gr.to_undirected()):
            conn_comp += 1
        #print(conn_comp)
    average_clustering = nx.average_clustering(gr)
    degree = nx.degree_centrality(gr)
    betweenness = nx.betweenness_centrality(gr)
    clustering = nx.clustering(gr)
    eigenvector = nx.eigenvector_centrality(gr)
    katz = nx.katz_centrality(gr)
    data = {'betweenness': betweenness, 'clustering': clustering, 'degree': degree, 'eigenvector': eigenvector,
            'katz': katz}
    df_new = pd.DataFrame.from_dict(data)
    return df_new

def military_civilian_mixing(graph):
    military_status = nx.get_node_attributes(graph, "military_status")
    mapping = {'Civilian': 0, 'Uniformed Service Member': 1, 'Department of Defense (DoD)': 2}
    mix_mat = nx.attribute_mixing_matrix(graph, 'military_status', mapping=mapping)
    print('Mixing value civilian to military (ignore dod): ')
    print(mix_mat[mapping['Civilian'], mapping['Uniformed Service Member']])
    # print(mix_mat)
    mapping = {'Civilian': 0, 'Uniformed Service Member': 1, 'Department of Defense (DoD)': 1}
    mix_mat = nx.attribute_mixing_matrix(graph, 'military_status', mapping=mapping)
    print('Mixing value civilian to (military and DOD): ')
    print(mix_mat[mapping['Civilian'], mapping['Uniformed Service Member']])
    # print(mix_mat)
    mapping = {'Civilian': 0, 'Uniformed Service Member': 1, 'Department of Defense (DoD)': 0}
    mix_mat = nx.attribute_mixing_matrix(graph, 'military_status', mapping=mapping)
    print('Mixing value (civ and DOD) to military: ')
    print(mix_mat[mapping['Civilian'], mapping['Uniformed Service Member']])
    # print(mix_mat)
    print()

def avg_analysis(G, G_noTB):
    analyses_with_TB: pd.DataFrame() = graph_analysis(G)
    # print(analyses_with_TB)
    average_analysis: dict = {}
    mean_betweenness = 0
    for i in analyses_with_TB.index:
        mean_betweenness += analyses_with_TB.at[i, 'betweenness']
    print('Mean betweenness centrality WITH TB:')
    print(mean_betweenness)
    analyses_no_TB: pd.DataFrame() = graph_analysis(G_noTB)
    # print(analyses_no_TB)
    average_analysis: dict = {}
    mean_betweenness = 0
    for i in analyses_no_TB.index:
        mean_betweenness += analyses_no_TB.at[i, 'betweenness']
    print('Mean betweenness centrality without TB:')
    print(mean_betweenness)

## MAIN #############################
# import json into a dataframe
#df_raw: pd.DataFrame = pd.read_json(r"C:\Users\megan\Desktop\Research\json_network_clean.txt", dtype="string")
df_raw: pd.DataFrame = pd.read_json(r"C:\Users\megan\PycharmProjects\pythonProject3\NWTB_DATA.json", dtype="string")


# df2 is a data frame,
# df2.section2 is a series of dicts
# df2.section2.items() are dicts of tuples
# each tuple is the index (int) and a dict
# section2 is a dict

df2: pd.DataFrame = concat_name(df_raw) #fist and last name become fullname

# separated the "section2" column to flatten the data
section2: pd.DataFrame = pd.DataFrame.from_dict(df2.section2)
section2 = section2['section2']

GG: nx.DiGraph = adding_company_and_mil_status_to_roster(df2)
GG = add_supervisors_pendants(df2, GG)

# Gfix = TB member separate but mutually connected
GG = attr_for_NWTB(GG)
# GG: nx.DiGraph = Graw
GG.remove_node('')

# THERE WAS SOME STATEMENTS OF THE FORM BELOW TO COMBINE TB MEMBERS 
# GG = nx.contracted_nodes(GG, 'node1', 'node2') 

# Adds attribute for company type, for the non-roster members.
GG = company_stuff(section2, GG)

GG = add_unknown_company_type(GG)
### PLOTTING ###
#plot_one_graph(GG_remove_TB)
# plot_combined(GG, GG_remove_TB)

print('Mixing WITH TB:')
military_civilian_mixing(GG)
print('Mixing WITHOUT TB:')
military_civilian_mixing(GG_remove_TB)
avg_analysis(GG, GG_remove_TB)
print()
# bar_graphs(graph_analysis(GG))
print('The number of nodes is:')
print(GG.number_of_nodes())
print('The number of survey takers is:')
print(len(df2.index))
print('The number of edges is: ')
print(GG.number_of_edges())
