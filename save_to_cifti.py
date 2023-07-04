# Import dependencies

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import scipy as sp
from itertools import combinations
import ciftools_FA as ct
import statsmodels.api as sm
from statsmodels.formula.api import ols
import subprocess as sp
import scikit_posthocs as ph
import networkx as nx
from joblib import Parallel, delayed
from settings import root_dir, output_dir, subj_id, group, lbl_N, nw_N, networks, NW_tbl, nw_name
subj_dir = f"{root_dir}/Subjects/"
gcca_dir = f"{output_dir}/GCCA"
#----------------------------------------------------------------------------------------------------


''' Save dscalar files'''

# Load gcca data
gcca_nw = np.load(f'{gcca_dir}/{group}.gcca.{nw_N}NWs.npy')
gcca_parc = np.load(f'{gcca_dir}/{group}.gcca.{lbl_N}Parc.npy')
gcca_vtx = np.load(f'{gcca_dir}/{group}.gcca.32k_fs_LR.npy')

# Get vertex network labels
_, lbl_tbl = ct.agg_networks(networks, networks, by_hemisphere=False, label_tbl=True)
vtx_nw = lbl_tbl.set_index('label').loc[networks.get_fdata().squeeze().astype('int32'), 'network'].values
template = nib.load(f'{subj_dir}/{subj_id[0]}/Analysis/{subj_id[0]}.GCCA_525.32k_fs_LR.dscalar.nii')

# Vertex - gradient median
scalars = np.median(gcca_vtx, axis=0)
out = f'{output_dir}/{group}.gcca.32k_fs_LR.dscalar.nii'
ct.save_dscalar(scalars, template, out, names=['Gradient_1', 'Gradient_2', 'Gradient_3'])

# Parcel - gradient median
scalars = np.median(gcca_parc[:, :, networks.get_fdata().squeeze().astype('int32')], axis=0)
out = f'{output_dir}/{group}.gcca.{lbl_N}Parc.32k_fs_LR.dscalar.nii'
ct.save_dscalar(scalars, networks, out, names=['Gradient 1', 'Gradient 2', 'Gradient 3'])

# Network - gradient median
scalars = np.median(gcca_nw[:, :, vtx_nw], axis=0)
out = f'{output_dir}/{group}.gcca.{lbl_N}Parc_{nw_N}NWs.32k_fs_LR.dscalar.nii'
ct.save_dscalar(scalars, networks, out, names=['Gradient_1', 'Gradient_2', 'Gradient_3'])


dispersion_nw = np.load(f'{output_dir}/{group}.gcca_dispersion.{nw_N}NWs.npy')
dispersion_parc = np.load(f'{output_dir}/{group}.gcca_dispersion.{lbl_N}Parc.npy')
dispersion_vtx = np.load(f'{output_dir}/{group}.gcca_dispersion.32k_fs_LR.npy')

# Vertex - dispersion 
scalars = dispersion_vtx
out = f'{output_dir}/{group}.gcca_dispersion.32k_fs_LR.dscalar.nii'
ct.save_dscalar(scalars, template, out, names=['All', 'Gradient_1', 'Gradient_2', 'Gradient_3'])

# Parcel - dispersion 
scalars = dispersion_parc[:, networks.get_fdata().squeeze().astype('int32')]
out = f'{output_dir}/{group}.gcca_dispersion.{lbl_N}Parc.32k_fs_LR.dscalar.nii'
ct.save_dscalar(scalars, networks, out, names=['All', 'Gradient_1', 'Gradient_2', 'Gradient_3'])

# Network - dispersion 
scalars = dispersion_nw[:, vtx_nw]
out = f'{output_dir}/{group}.gcca_dispersion.{lbl_N}Parc_{nw_N}NWs.32k_fs_LR.dscalar.nii'
ct.save_dscalar(scalars, networks, out, names=['All', 'Gradient_1', 'Gradient_2', 'Gradient_3'])
