# Import dependencies

import nibabel as nib
import numpy as np
import ciftools_FA as ct
from settings import *
#----------------------------------------------------------------------------------------------------


''' Calculate dispersion of connectivity in 3D gradient space'''

gcca_nw = np.load(f'{gcca_dir}/{group}.gcca.{nw_N}NWs.npy')
gcca_parc = np.load(f'{gcca_dir}/{group}.gcca.{lbl_N}Parc.npy')
gcca_vtx = np.load(f'{gcca_dir}/{group}.gcca.32k_fs_LR.npy')

# AVVERAGE SQUARED DISTANCE from group centroid
def gcca_dispersion(gcca):
    centroids = gcca.mean(axis=0)
    squares = np.square(gcca - centroids)
    sum_squares = squares[:, :2, :].sum(axis=1)
    distance = np.sqrt(sum_squares)
    avg_distance2 = np.square(distance).sum(axis=0) / distance.shape[0]
    avg_components2 = squares.sum(axis=0) / squares.shape[0]
    return np.vstack([avg_distance2, avg_components2])

dispersion_nw = gcca_dispersion(gcca_nw)
dispersion_parc = gcca_dispersion(gcca_parc)
dispersion_vtx = gcca_dispersion(gcca_vtx)


np.save(f'{output_dir}/{group}.gcca_dispersion.{nw_N}NWs', dispersion_nw)
np.save(f'{output_dir}/{group}.gcca_dispersion.{lbl_N}Parc', dispersion_parc)
np.save(f'{output_dir}/{group}.gcca_dispersion.32k_fs_LR', dispersion_vtx)
#----------------------------------------------------------------------------------------------------



''' Save dscalar files'''

# Load gcca data
gcca_nw = np.load(f'{gcca_dir}/{group}.gcca.{nw_N}NWs.npy')
gcca_parc = np.load(f'{gcca_dir}/{group}.gcca.{lbl_N}Parc.npy')
gcca_vtx = np.load(f'{gcca_dir}/{group}.gcca.32k_fs_LR.npy')

# Get vertex network labels
_, lbl_tbl = ct.agg_networks(networks, networks, by_hemisphere=False, label_tbl=True)
vtx_nw = lbl_tbl.set_index('label').loc[networks.get_fdata().squeeze().astype('int32'), 'network'].values
template = nib.load(f'{gcca_dir}/{subj_id[0]}.GCCA.32k_fs_LR.dscalar.nii')

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




