"""Add most commonly used graph metrics as graph and nodes attributes"""

import networkx as nx
import numpy as np

__all__ = [
    "full_diagnostics"
]

def full_diagnostics(G, modules=None, swi=False, swi_niter=100, swi_nrand=10, swi_seed=None, n_jobs=None, prefer=None):    
    
    # swi = False
    # Global metrics
    G.metrics = {}
    
    connected = nx.is_connected(G)
    G.metrics[f"connected"] = connected
    
    if modules is None:
        modules = nx.algorithms.community.louvain_communities(G)        
        
    modularity = nx.algorithms.community.modularity(G, modules, weight='weight')
    G.metrics[f"modularity"] = modularity
    
    global_e = nx.algorithms.efficiency_measures.global_efficiency(G, avg=True)  # original fun edited to return dict
    G.metrics[f"global_e"] = global_e
    
    local_e = nx.algorithms.efficiency_measures.local_efficiency(G, avg=True)  # original fun edited to return dict
    G.metrics[f"local_e"] = local_e
    
    clustering_avg = nx.average_clustering(G, weight='weight')
    G.metrics[f"clustering"] = clustering_avg
    
    percentage_connected = np.max(list(len(c)/len(G) for c in nx.connected_components(G)))
    G.metrics[f"connected_pct"] = clustering_avg
    
    if swi:
        sigma = nx.sigma(G, niter=swi_niter, nrand=swi_nrand, seed=swi_seed)
        G.metrics[f"SWI"] = sigma
    else:
        G.metrics[f"SWI"] = np.nan
     
    
    # Nodal metrics
    strength = nx.degree(G, weight='weight')
    nx.set_node_attributes(G, dict(strength), name=f'strength')
    
    if connected:
        eccentricity = nx.eccentricity(G)                                  #Issues with connectedness
        nx.set_node_attributes(G, eccentricity, name=f'eccentricity')
    else:
        nx.set_node_attributes(G, np.nan, name=f'eccentricity')
        
    global_e = nx.algorithms.efficiency_measures.global_efficiency(G, avg=False)  # original fun edited to return dict
    nx.set_node_attributes(G, global_e, name=f'global_e')
    
    local_e = nx.algorithms.efficiency_measures.local_efficiency(G, avg=False)  # original fun edited to return dict
    nx.set_node_attributes(G, local_e, name=f'local_e')
    
    clustering = nx.clustering(G, weight='weight')
    nx.set_node_attributes(G, clustering, name=f'clustering')
    
    degree_c = nx.degree_centrality(G)
    nx.set_node_attributes(G, degree_c, name=f'degree_c')
    
    between_c = nx.betweenness_centrality(G, weight='weight')
    nx.set_node_attributes(G, between_c, name=f'between_c')
    
    participation = nx.algorithms.participation_coef(G, modules)
    nx.set_node_attributes(G, participation, name=f'participation_c')
    
    # eigen_c = nx.eigenvector_centrality(G)
    # nx.set_node_attributes(G, eigen_c, name=f'eigen_c')
    
    for i, mod in enumerate(modules):
        attr = dict(zip(modules[i], np.repeat(i, len(mod))))
        nx.set_node_attributes(G, attr, name=f'module')
    
    return G