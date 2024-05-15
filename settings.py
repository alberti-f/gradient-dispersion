# Customizable settings

''' RECURRING VARIABLES '''

import numpy as np
import nibabel as nib
import ciftools_FA as ct



# group name to prefix the output files
group = "338" 

# directory of HCP group average data
group_dir = "/home/fralberti/Data/HCP_1200/HCP_S1200_GroupAvg_v1"

# directory containing the HCP subject directories
subj_dir = "home/fralberti/Data/HCP_1200/Subjects"

# directory where all intermediate files and the final output will be saved
output_dir = f"/home/fralberti/Documents/BlackBox/Prj_Gradient-Variability/Results_{group}"
# directory containing the GCCA results
gcca_dir = f"/path/to/GCCA/results"

# path to cognitive data
cog_path = "/path/to/unrestricted/data.csv"

# path to restricted data
rest_path = "/path/to/restricted/data.csv"

# .txt containing IDs of subjects to include in the analysis
subj_list = '/path/to/subject_list.txt'
f = open(subj_list, 'r')
subj_id = np.array(f.read().splitlines(), dtype='int32')


# Schaefer atlas to use for analyses
lbl_N = 400
nw_N = 7

# Path to Shaefer2018 atlas registered to HCP
networks_dir = f'/path/to/Shaefer2018_HCP'
networks_txt = f'{networks_dir}/Schaefer2018_{lbl_N}Parcels_{nw_N}Networks_order_info.txt'
networks = nib.load(f'{networks_dir}/Schaefer2018_{lbl_N}Parcels_{nw_N}Networks_order.dlabel.nii')
NW_tbl = ct.agg_networks(networks, networks, func=np.median, by_hemisphere=False, label_tbl=True)[1]
nw_name = NW_tbl.groupby('name').agg(np.median).sort_values('network').index.to_list()[1:]

# number of jobs for parallelized operations
nj = -1
