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
from settings import root_dir, output_dir, subj_id, group, lbl_N, nw_N, networks, networks_txt, NW_tbl, nw_name, nj
subj_dir = f"{root_dir}/Subjects/"
gcca_dir = f"{output_dir}/GCCA"
#----------------------------------------------------------------------------------------------------



''' Generate clusters of highest inter-individual variability '''

# Find variance clusters
dscalar = f"{output_dir}/{group}.gcca_dispersion.32k_fs_LR.dscalar.nii"
percentile = 95
clusters = []
for g_tmp in range(4):
    thr = np.percentile(np.load(f"{output_dir}/{group}.gcca_dispersion.32k_fs_LR.npy"), percentile, axis=1)[g_tmp]
    srf_min = 200
    vol_min = 1
    out = f"{output_dir}/{group}.gcca_disp_clusters.32k_fs_LR.dscalar.nii"
    L_srf = f"{root_dir}/HCP_S1200_GroupAvg_v1/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
    R_srf = f"{root_dir}/HCP_S1200_GroupAvg_v1/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii"

    sp.run(f"wb_command -cifti-find-clusters {dscalar} {thr} {srf_min} {thr} {vol_min} COLUMN {out} -left-surface {L_srf} -right-surface {R_srf}", shell=True)

    clusters_g = nib.load(f"{output_dir}/{group}.gcca_disp_clusters.32k_fs_LR.dscalar.nii").get_fdata()[g_tmp, :]
    clusters.append(clusters_g)

clusters = np.asarray(clusters)
clusters = stats.rankdata(clusters, axis=1, method='dense') - 1

ct.save_dscalar(clusters, nib.load(dscalar), out, names=['All', 'Gradient_1', 'Gradient_2', 'Gradient_3'])
#----------------------------------------------------------------------------------------------------



''' Save clusters as dlabel.nii file '''

clusters = nib.load(f"{output_dir}/{group}.gcca_disp_clusters.32k_fs_LR.dscalar.nii")


labels_path = f"{output_dir}/{group}.gcca_disp_clusters.32k_fs_LR.dscalar.nii"
labels_array = nib.load(labels_path)
lbl_txt =  f'{output_dir}/{group}.gcca_disp_clusters_info.txt'
new_dlabel = f'{output_dir}/{group}.gcca_disp_clusters_info.dlabel.txt'


labels = np.unique(clusters.get_fdata()).astype('int32')
new_labels = []
colors = np.round(plt.cm.Dark2(range(len(labels)-1))* 255).astype('int32').astype('str')
for lbl in labels[labels!=0]:
    new_labels.append(f"Disp_cluster_{lbl}")
    new_labels.append(' '.join(np.hstack([lbl, colors[lbl-1,:]])))

#with open(lbl_txt, 'w') as f:
#    #lines = f.readlines()
#    lines.extend([s + '\n' for s in new_labels])
#    f.close()

with open(lbl_txt, 'w') as f:
    f.write('\n'.join(new_labels))
    
    
# Create dlabel file
new_dlabel = f'{output_dir}/{group}.gcca_disp_clusters.32k_fs_LR.dlabel.nii'
cmd = f'wb_command -cifti-label-import {labels_path} {lbl_txt} {new_dlabel} -discard-others'
sp.run(cmd, shell=True)

#----------------------------------------------------------------------------------------------------



''' Add dispersion cluster to Schaefer atlas '''

clusters = nib.load(f"{output_dir}/{group}.gcca_disp_clusters.32k_fs_LR.dscalar.nii")

# allign network and cluster maps
shared_l = np.isin(ct.struct_info('CIFTI_STRUCTURE_CORTEX_LEFT', networks)[2], ct.struct_info('CIFTI_STRUCTURE_CORTEX_LEFT', clusters)[2])
shared_r = np.isin(ct.struct_info('CIFTI_STRUCTURE_CORTEX_RIGHT', networks)[2], ct.struct_info('CIFTI_STRUCTURE_CORTEX_RIGHT', clusters)[2])
shared_vv = np.hstack([shared_l, shared_r])

clusters = clusters.get_fdata()[0].astype('int32')
clusters[clusters != 0] += 1000    ### add 1000 to differentiate from original parcels
cluster_N = len(np.unique(clusters))
parcels = networks.get_fdata()[0, shared_vv].astype('int32')
parcels[clusters!=0] = clusters[clusters!=0]
removed = np.unique(networks.get_fdata()[0, shared_vv].astype('int32'))[~np.isin(np.unique(networks.get_fdata()[0, shared_vv].astype('int32')), parcels)]

# save dscalar with new parcellation
out = f'{output_dir}/{group}.dispROIs_Schaefer2018_{lbl_N}Parcels_{nw_N}Networks.dscalar.nii'
template = nib.load(f"{output_dir}/{group}.gcca_disp_clusters.32k_fs_LR.dscalar.nii")
ct.save_dscalar(parcels, template, out)


# Create list of new parcels and colors
new_labels = []
names = [f"{nw_N}Networks_LH_ROI_{i}" for i in range(1,cluster_N)]
colors = np.round(plt.cm.Dark2(range(np.unique(clusters[clusters!=0]).size))* 255).astype('int32').astype('str')
for i, lbl in enumerate(np.unique(clusters[clusters!=0])):
    new_labels.append(names[i]+'\n')
    new_labels.append(' '.join(np.hstack([lbl, colors[i,:], '\n'])))
   
    
# Save new label-list-file with added parcels
orig_lbl_txt = networks_txt
with open(orig_lbl_txt, 'r+') as f:
    lines = f.readlines()
    lines.extend(new_labels)
    f.close()

for i in range(1, len(lines), 2):
    if i > len(lines):
        break
    label = int(lines[i].split()[0])
    if label not in parcels:
        del lines[i], lines[i-1]

new_lbl_txt = f'{output_dir}/{group}.dispROIs_Schaefer2018_{lbl_N}Parcels_{nw_N}Networks.txt'
with open(new_lbl_txt, 'w') as f:
    f.truncate(0)
    f.write(''.join(lines))
    f.close()
    
# Create dlabel file
new_dlabel = f'{output_dir}/{group}.dispROIs_Schaefer2018_{lbl_N}Parcels_{nw_N}Networks.dlabel.nii'
cmd = f'wb_command -cifti-label-import {out} {new_lbl_txt} {new_dlabel}'
sp.run(cmd, shell=True)

#----------------------------------------------------------------------------------------------------



''' Generate previous tables and images using the integrated parcellation '''

joint_atlas = nib.load(f'{output_dir}/{group}.dispROIs_Schaefer2018_{lbl_N}Parcels_{nw_N}Networks.dlabel.nii')
labels = np.unique(joint_atlas.get_fdata()).astype('int32')

# Parcel - gradient median
gcca_parc = []
for subj in subj_id:
    grad = nib.load(f'{subj_dir}/{subj}/Analysis/{subj}.GCCA_525.32k_fs_LR.dscalar.nii')
    lbl_df = ct.agg_labels(grad, joint_atlas, func=np.median)
    gcca_parc.append(lbl_df.T.values)
    np.save(f'{output_dir}/{subj}.gcca.dispROIs_{lbl_N}Parc', lbl_df)
    
gcca_parc = np.array(gcca_parc)
np.save(f'{output_dir}/{group}.gcca.dispROIs_{lbl_N}Parc', gcca_parc)

parc_idx = stats.rankdata(joint_atlas.get_fdata().squeeze(), method='dense') - 1
scalars = np.median(gcca_parc[:, :, parc_idx], axis=0)
out = f'{output_dir}/{group}.gcca.dispROIs_{lbl_N}Parc.32k_fs_LR.dscalar.nii'
ct.save_dscalar(scalars, joint_atlas, out, names=['Gradient 1', 'Gradient 2', 'Gradient 3'])


# Parcel - dispersion 
dispersion_vtx = nib.load(f'{output_dir}/{group}.gcca_dispersion.32k_fs_LR.dscalar.nii')
dispersion_parc = ct.agg_labels(dispersion_vtx, joint_atlas, func=np.median)
np.save(f'{output_dir}/{group}.gcca_dispersion.dispROIs_{lbl_N}Parc', dispersion_parc)

scalars = joint_atlas.get_fdata().copy()

for lbl, val in dispersion_parc.iterrows():
    scalars[:,scalars[0]==lbl] = val.iloc[0]
    
out = f'{output_dir}/{group}.gcca_dispersion.dispROIs_{lbl_N}Parc.32k_fs_LR.dscalar.nii'
ct.save_dscalar(scalars, joint_atlas, out, names=['All'])


