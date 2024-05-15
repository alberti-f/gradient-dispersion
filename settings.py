# Customizable settings

''' RECURRING VARIABLES '''

import numpy as np
import nibabel as nib
import ciftools_FA as ct
import pandas as pd

# directory containing subdirectories named fter subject IDs that contain the timeseries and surface files
root_dir = "/home/fralberti/Data/HCP_1200"
# directory where all intermediate files and the final output will be saved
output_dir = "/home/fralberti/Documents/BlackBox/Prj_Network-variability/Results"
# path to cognitive data
cog_path = f'{root_dir}/unrestricted_fralberti_4_8_2022_8_22_23.csv'
# .txt containing IDs of subjects to include in the analysis
f = open(f'{root_dir}/Subjects/subj_IDs_338.txt', 'r')
subj_id = np.array(f.read().splitlines(), dtype='int32')
# group name
group = '338'

# Schaefer atlas to use for analyses
lbl_N = 400
nw_N = 7
networks_txt = f'/home/fralberti/Data/Shaefer2018_HCP/Schaefer2018_{lbl_N}Parcels_{nw_N}Networks_order_info.txt'
networks = nib.load(f'/home/fralberti/Data/Shaefer2018_HCP/Schaefer2018_{lbl_N}Parcels_{nw_N}Networks_order.dlabel.nii')
NW_tbl = ct.agg_networks(networks, networks, func=np.median, by_hemisphere=False, label_tbl=True)[1]
nw_name = NW_tbl.groupby('name').agg(np.median).sort_values('network').index.to_list()[1:]

# number of jobs for parallelized operations
nj = -1
