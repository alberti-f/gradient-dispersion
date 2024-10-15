# Customizable settings

''' RECURRING VARIABLES '''

import numpy as np
import nibabel as nib
import ciftools_FA as ct
import pandas as pd



# group name to prefix the output files
group = "group_name" 

# directory of HCP group average data
group_dir = "/path/to/HCP_S1200_GroupAvg_v1"

# directory containing the HCP subject directories
subj_dir = "/path/to/HCP/subject/directories"

# directory where all intermediate files and the final output will be saved
output_dir = "/path/to/output/directory"

# directory containing the GCCA results
gcca_dir ="/path/to/GCCA/results"


# path to cognitive data
cog_path = "/path/to/unrestricted/data.csv"

# path to restricted data
rest_path = "/path/to/RESTRICTED/data.csv"

# .txt containing IDs of subjects to include in the analysis

subj_list = '/path/to/subject_list.txt'
f = open(subj_list, 'r')
subj_id = np.array(f.read().splitlines(), dtype='int32')


# Schaefer atlas to use for analyses
lbl_N = 400
nw_N = 7

# Path to Shaefer2018 atlas registered to HCP
networks_dir = '/path/to/Shaefer2018_HCP'
networks_txt = f'{networks_dir}/Schaefer2018_{lbl_N}Parcels_{nw_N}Networks_order_info.txt'
networks = nib.load(f'{networks_dir}/Schaefer2018_{lbl_N}Parcels_{nw_N}Networks_order.dlabel.nii')
NW_tbl = ct.agg_networks(networks, networks, func=np.median, by_hemisphere=False, label_tbl=True)[1]
NW_tbl[["r", "g", "b", "a"]] = pd.DataFrame(NW_tbl['rgba'].tolist(), index=NW_tbl.index)
NW_tbl = NW_tbl.drop(columns=['rgba'])
nw_name = NW_tbl.groupby('name').agg(np.median).sort_values('network').index.to_list()[1:]

# number of jobs for parallelized operations
nj = -1

radius = 50.0
rest_FD = []
for subj in subj_id:
    # Compute FD of traslations
    runs = ["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"]
    subj_FD = []
    for i, run in enumerate(runs):
        traslations = np.loadtxt(f"{subj_dir}/{subj}/MNINonLinear/Results/rfMRI_{run}/Movement_Regressors.txt",
                                       usecols=[0,1,2], dtype="float64")
        translation_deltas = np.abs(traslations).sum(axis=1)

        rotations = np.loadtxt(f"{subj_dir}/{subj}/MNINonLinear/Results/rfMRI_{run}/Movement_Regressors.txt",
                                       usecols=[3,4,5], dtype="float64")
        rotations_deltas = (abs(rotations) / 360) * (2 * np.pi * radius)
        rotations_deltas = rotations_deltas.sum(axis=1)

        FD = translation_deltas + rotations_deltas
        subj_FD.extend(FD)

    rest_FD.append(subj_FD)

if np.any(np.mean(rest_FD,axis=1) > 25):
    print("WARNING: High motion subjects:")
    ids = subj_id[np.where(np.median(rest_FD,axis=1) > 25)]
    fd = np.mean(rest_FD,axis=1)[np.where(np.mean(rest_FD,axis=1) > 25)]
    print("\n".join([f"{i}: {f:.2f}" for i, f in zip(ids, fd)]))
    print("These subjects should be excluded from the analysis")
else:
    print("No high motion subjects")
    print("Group average FD: ", np.mean(np.median(rest_FD,axis=1)))
    print("Group FD range: ", np.min(np.median(rest_FD,axis=1)), np.max(np.median(rest_FD,axis=1)))