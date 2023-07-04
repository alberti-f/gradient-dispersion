# Import dependencies

import os
import nibabel as nib
import numpy as np
from scipy import stats
import pandas as pd
import ciftools_FA as ct
import statsmodels.api as sm
from joblib import Parallel, delayed
from settings import root_dir, output_dir, subj_id, group, lbl_N, nw_N, networks, NW_tbl, nw_name, nj
subj_dir = f"{root_dir}/Subjects/"
gcca_dir = f"{output_dir}/GCCA"
#----------------------------------------------------------------------------------------------------

''' Save vertex/parcel/network-level GCCA (individual and group median) from dscalar.nii to .npy arrays'''

# Write function to run in parallel
def fun(subj, root_dir, output_dir, nw_N, lbl_N):
    grad = nib.load(f'{root_dir}/{subj}/Analysis/{subj}.GCCA_525.32k_fs_LR.dscalar.nii')
    NW_df, NW_tbl = ct.agg_networks(grad, networks, func=np.median, by_hemisphere=False, label_tbl=True)
    NW_df.iloc[0, :] = 0
    NW_tbl.iloc[0, :] = 0
    lbl_df = ct.agg_labels(grad, networks, func=np.median)    
    
    # Append subject's vertex, label, and network-level gradients to group list (for SD)
    grad_NW = NW_df.T.values
    grad_lbl = lbl_df.T.values
    grad_vtx = np.asarray(grad.get_fdata())
    
    # Save median individual median gradient of parcels and networks
    np.save(f'{output_dir}/{subj}.gcca.{nw_N}NWs', NW_df) 
    np.save(f'{output_dir}/{subj}.gcca.{lbl_N}Parc', lbl_df)
    
    return grad_NW, grad_lbl, grad_vtx
    
#--------------------------------------------------------------------------------------------------

# Create directory

if not os.path.exists(gcca_dir):
	os.mkdir(gcca_dir, mode = 0o777)

r = Parallel(n_jobs=nj)(delayed(fun)(subj, subj_dir, gcca_dir, nw_N, lbl_N) for subj in subj_id)
# unpack output
grad_NW, grad_lbl, grad_vtx = zip(*r)

# Save parcels' and networks' median gradient of all subjects
np.save(f'{gcca_dir}/{group}.gcca.{nw_N}NWs', np.array(grad_NW)) 
np.save(f'{gcca_dir}/{group}.gcca.{lbl_N}Parc', np.array(grad_lbl))
np.save(f'{gcca_dir}/{group}.gcca.32k_fs_LR', np.array(grad_vtx)) 

