"""Graph participation coefficient"""

import networkx as nx
import numpy as np

__all__ = [
    "participation_coef",
    "single_node_community_SPL",
    "node_community_SPL",
    "node_community_degree"
]


def single_node_community_SPL(G, source, community, normalize=False, weight="weight"):
    SPLs = nx.shortest_path_length(G, source, weight=weight)
    del SPLs[source]
    neighbors_in_comm = set(community) & set(SPLs.keys())
    
    if not neighbors_in_comm:
        node_comm_SPL = np.inf     
    else:
        vals = np.asarray(list(SPLs.values()))
        u = vals.mean()
        sd =vals.std()
        node_comm_SPL = np.asarray([SPLs[n] for n in neighbors_in_comm])
        if normalize:
            node_comm_SPL = (node_comm_SPL.mean() - u) / sd
        else:
            node_comm_SPL = node_comm_SPL.mean()
        
    return node_comm_SPL



def node_community_SPL(G, community, normalize=False, weight="weight"):
    SPLs = map(lambda n: nx.algorithms.single_node_community_SPL(G, n, community, normalize=normalize, weight=weight), G)
    return dict(zip(G, SPLs))
    
    

def node_community_degree(G, community, weight="weight"):
    m = map(lambda n: sum([G[n].get(i)[weight] for i in community if i in G.neighbors(n)]), G.nodes)
    return dict(zip(G.nodes, m))
    


def participation_coef(G, communities, weight="weight"):
    
    n = len(G)
    Ko = dict(nx.degree(G, weight=weight)) #degree/sum of weights            
    Kc2 = np.zeros(len(G))
    for c in communities:
        Kc2 += np.asarray(list(node_community_degree(G, c, weight=weight).values())) ** 2
    Kc2 = dict(zip(G.nodes, Kc2))
    
    P = {n:1 - Kc2[n] /  Ko[n] ** 2 if Ko[n]>0 else 0 for n in G.nodes}
    return P
