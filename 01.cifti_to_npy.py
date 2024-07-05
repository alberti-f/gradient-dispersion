# Import dependencies

import os
import nibabel as nib
import numpy as np
import ciftools_FA as ct
from joblib import Parallel, delayed
from settings import * 
#----------------------------------------------------------------------------------------------------



''' Save vertex/parcel/network-level GCCA (individual and group median) from dscalar.nii to .npy arrays'''

template = nib.load(f"{group_dir}/S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii")

# Write function to run in parallel
def fun(subj, output_dir, nw_N, lbl_N):
    """
    Process individual gradients and save them to files.

    Parameters:
    - subj (int): Subject identifier.
    - output_dir (str): Directory to save the output files.
    - nw_N (int): Number of networks.
    - lbl_N (int): Number of parcels.

    Returns:
    - grad_NW (ndarray): Array of subject's network-level gradients.
    - grad_lbl (ndarray): Array of subject's label-level gradients.
    - grad_vtx (ndarray): Array of subject's vertex-level gradients.
    """

    # Save individual gradients to cifti
    out_grad = f"{gcca_dir}/{subj}.GCCA.32k_fs_LR.dscalar.nii"
    grad = np.load(f"{gcca_dir}/{subj}.GCCA.npy")
    ct.save_dscalar(grad, template, out_grad, names=['Gradient_1', 'Gradient_2', 'Gradient_3'])

    # Aggregate gradients by parcels and networks
    grad = nib.load(out_grad)
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
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

r = Parallel(n_jobs=nj)(delayed(fun)(subj, gcca_dir, nw_N, lbl_N) for subj in subj_id)
# unpack output
grad_NW, grad_lbl, grad_vtx = zip(*r)

# Save parcels' and networks' median gradient of all subjects
np.save(f'{gcca_dir}/{group}.gcca.{nw_N}NWs', np.array(grad_NW)) 
np.save(f'{gcca_dir}/{group}.gcca.{lbl_N}Parc', np.array(grad_lbl))
np.save(f'{gcca_dir}/{group}.gcca.32k_fs_LR', np.array(grad_vtx)) 


from neuromaps import images, nulls, stats, datasets
import nibabel as nib
import hcp_utils as hcp



# Test correlation with DME gradients

print("Comparing GCCA gradients with DME gradients: ")
pairs = [(1, 1), (3, 2), (2, 3)]
for n1, n2 in pairs:

    grad_lr = datasets.fetch_annotation(source="margulies2016", desc=f"fcgradient0{n1}")
    grad_lr = images.load_data(grad_lr)

    gcca_lr = nib.load(f'{gcca_dir}/{group}.gcca.32k_fs_LR').get_fdata()
    gcca_lr = hcp.cortex_data(gcca_lr[n2-1])
    rotated = nulls.alexander_bloch(gcca_lr, atlas='fsLR', density='32k',
                                    n_perm=1000)


    corr, pval = stats.compare_images(gcca_lr, grad_lr, nulls=rotated)
    print(f'\tDME: {n1} - GCCA: {n2} \t r = {corr:.02f}, p = {pval:.04f}')

#--------------------------------------------------------------------------------------------------