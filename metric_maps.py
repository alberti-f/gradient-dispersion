# Import dependencies

import os
import nibabel as nib
import numpy as np
from scipy import stats
import pandas as pd
import ciftools_FA as ct
import statsmodels.api as sm
import matplotlib.pyplot as plt
import subprocess as sp
from joblib import Parallel, delayed
import networkx as nx
from settings import root_dir, output_dir, subj_id, group, lbl_N, nw_N, networks, networks_txt, NW_tbl, nw_name, nj
subj_dir = f"{root_dir}/Subjects/"
gcca_dir = f"{output_dir}/GCCA"
#----------------------------------------------------------------------------------------------------


''' Test correlation between dispersion of graph metrics and principal gradient '''

gcca_dispersion = np.load(f'{output_dir}/{group}.gcca_dispersion.dispROIs_{lbl_N}Parc.npy')[:,1:]
gcca_median = np.median(np.load(f'{output_dir}/{group}.gcca.dispROIs_{lbl_N}Parc.npy')[:,0,1:].squeeze(), axis=0)
graph_dispersion_df = pd.read_csv(f'{output_dir}/{group}.graph_metrics_dispersion.IQR_Schaefer2018_{lbl_N}Parc_{nw_N}NW.csv')

graph_dispersion_df['gcca_disp'] = gcca_dispersion[1:,0]
graph_dispersion_df['gcca'] = gcca_median
graph_dispersion_df[graph_dispersion_df==0] = np.nan


results = {}

for i, graph_dispersion in enumerate(graph_dispersion_df.T.iterrows()):
    if i == 6:
        break
    x = graph_dispersion_df['gcca_disp']
    y = graph_dispersion[1]
    spearman_r = stats.spearmanr(x, y, nan_policy='omit')
    
    results[graph_dispersion[0]] = {'r': spearman_r[0],
                                    'p': spearman_r[1]}

p_unadj = [value['p'] for key, value in results.items()]
p_adj = sm.stats.multipletests(p_unadj, method='fdr_bh')[1]

for i, key in enumerate(results):
    results[key]['p_adj'] = p_adj[i]
    
print(pd.DataFrame(results).T)

#----------------------------------------------------------------------------------------------------


''' Test cross-subject correlation between graph metrics and principal gradient '''

attributes = ['strength', 'clustering', 'global_e', 'local_e', 'between_c', 'participation_c']


gcca = np.load(f'{output_dir}/{group}.gcca.dispROIs_{lbl_N}Parc.npy')[:,:,1:]
gmetrics = np.load(f'{output_dir}/{group}.graph_metrics.dispROIs_{lbl_N}Parc.npy')

results = {attribute:{'r':[], 'p':[]} for attribute in attributes}

for i, parc_metrics in enumerate(gmetrics[:, :, :].T):
    for j, metric in enumerate(parc_metrics):
        r, p = stats.spearmanr(metric, gcca[:,0,i], nan_policy='omit')
        results[attributes[j]]['r'].extend(np.array([r]))
        results[attributes[j]]['p'].extend(np.array([p]))
        
for metric, dictionary in results.items():
    h, p, _, _ = sm.stats.multipletests(dictionary['p'], alpha=0.05, method='fdr_by')
    dictionary['p'] = p
    dictionary['r'] = dictionary['r']
    
scalars = [results[metric][index] for metric in results.keys() for index in results[metric].keys()]
scalars = np.array(scalars)


joint_atlas = nib.load(f'{output_dir}/{group}.dispROIs_Schaefer2018_{lbl_N}Parcels_{nw_N}Networks.dlabel.nii')
labels = joint_atlas.get_fdata().copy().squeeze().astype('int32')
label_idx = stats.rankdata(labels, method='dense').squeeze() -1 

maps_array = np.zeros([scalars.shape[0], labels.size])

for i, scalar in enumerate(scalars):
    for lbl in np.unique(label_idx):
        if lbl==0:
            continue
        maps_array[i, label_idx==lbl] = scalar[lbl-1]
    
out = f'{output_dir}/{group}.graph_metric_correlation.dispROIs_{lbl_N}Parc.32k_fs_LR.dscalar.nii'
names = np.asanyarray([[f'{metric}_R', f'{metric}_pval'] for metric in results]).reshape(-1).tolist()
ct.save_dscalar(maps_array, joint_atlas, out, names=names)





