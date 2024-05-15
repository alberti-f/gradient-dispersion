# Import dependencies

import nibabel as nib
import numpy as np
from scipy import stats
import pandas as pd
import ciftools_FA as ct
from joblib import Parallel, delayed
import networkx as nx
from settings import *
import sys

if len(sys.argv) == 1:
    atlas_path = f'{output_dir}/{group}.dispROIs_Schaefer2018_{lbl_N}Parcels_{nw_N}Networks.dlabel.nii'
else:
    atlas_path = sys.argv[1]
#----------------------------------------------------------------------------------------------------


''' Generate FC graphs and extract topological metrics '''

thr = 90

NWs_n_ROIs = nib.load(atlas_path)

modules_df = ct.agg_networks(NWs_n_ROIs, NWs_n_ROIs)[1][1:].reset_index()

communities = [set(modules_df.label.astype('int32')[modules_df.network==nw].values) for nw in modules_df.network.unique()]
nodes = dict( zip( range( len(modules_df.index) ), modules_df.label) )

def graph_diagnostics(subj, communities, thr):
    M = np.genfromtxt(f'{subj_dir}/{subj}/Analysis/{subj}.REST_All_fcMatrix.csv', delimiter=',')[1:, 1:]
    M[M<np.percentile(M, thr)] = 0
    G = nx.from_numpy_array(M)
    G = nx.relabel_nodes(G, nodes)
    G = nx.algorithms.full_diagnostics(G, modules=communities, swi=False)
    nx.write_gpickle(G,f'{subj_dir}/{subj}/Analysis/{subj}.rfMRI_graph_{thr}.Schaefer_400parcs.gpickle')
    return [subj]

_ = Parallel(n_jobs=nj, prefer='processes')(delayed(graph_diagnostics)(subj, communities, thr) for subj in subj_id)

#----------------------------------------------------------------------------------------------------


''' Save graph metrics of interest to cifti '''

attributes = ['strength', 'clustering', 'global_e', 'local_e', 'between_c', 'participation_c']

def get_graph(subj):
    return nx.read_gpickle(f'{subj_dir}/{subj}/Analysis/{subj}.rfMRI_graph_{thr}.Schaefer_400parcs.gpickle')

individual_maps = []
for subj in subj_id:
    G = get_graph(subj)
    metrics = np.array([list(nx.get_node_attributes(G, attribute).values()) for attribute in attributes])
    
    isolates = nx.isolates(G)
    metrics[:,np.isin(G, list(isolates))] = np.nan
    individual_maps.append(metrics)
    
    
individual_maps = np.array(individual_maps)
np.save(f'{output_dir}/{group}.graph_metrics.dispROIs_{lbl_N}Parc.npy', individual_maps)

graph_dispersion_maps = stats.iqr(individual_maps, 0, rng=(25, 75), nan_policy='omit') 
graph_dispersion_df = pd.DataFrame(graph_dispersion_maps.T, columns=attributes)
graph_dispersion_df.to_csv(f'{output_dir}/{group}.graph_metrics_dispersion.IQR_Schaefer2018_{lbl_N}Parc_{nw_N}NW.csv', index=False)


joint_atlas = nib.load(atlas_path)
labels = joint_atlas.get_fdata().copy().squeeze().astype('int32')
label_idx = stats.rankdata(labels, method='dense').squeeze() -1 


scalars = np.zeros([graph_dispersion_df.shape[1], labels.size]).T

for lbl, val in graph_dispersion_df.iterrows():
    scalars[label_idx==lbl+1, :] = val.values

out = f'{output_dir}/{group}.graph_dispersion.dispROIs_{lbl_N}Parc.32k_fs_LR.dscalar.nii'
ct.save_dscalar(scalars.T, joint_atlas, out, names=attributes)

#---------------------------------------------------------------------------------------------------
