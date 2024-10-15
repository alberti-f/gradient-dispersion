# Import dependencies

import nibabel as nib
import numpy as np
import pandas as pd
import ciftools_FA as ct
from settings import *
import sys


if len(sys.argv) == 1:
    clusters_path = f"{output_dir}/{group}.gcca_disp_clusters.32k_fs_LR.dscalar.nii"
else:
    clusters_path = sys.argv[1]

#---------------------------------------------------------------------------------------------------

# Generate table with vertex affiliations, gradients, and dispersion measures

grad_SD = nib.load(f"{output_dir}/{group}.gcca_dispersion.32k_fs_LR.dscalar.nii")
cifti_59k = nib.load(f"{gcca_dir}/{subj_id[0]}.GCCA.32k_fs_LR.dscalar.nii")

# Create parcel label array
parc_lbl = []
vv = []
hemi = []
for struct in ["CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"]:
    NWos, NWn, NWvv = ct.struct_info(struct, networks)
    Gos, Gn, Gvv = ct.struct_info(struct, cifti_59k)
    shared_vv = np.isin(NWvv, Gvv)
    parc_lbl.extend(networks.get_fdata()[0, NWos : NWos + NWn][shared_vv])
    vv.extend(NWvv[shared_vv])
    h = (0 if struct=="CIFTI_STRUCTURE_CORTEX_LEFT" else 1)
    hemi.extend(np.repeat(h, Gn))


# Create network label array
parc_to_NW = ct.agg_networks(networks, networks, func="mean", by_hemisphere=False, label_tbl=True)[1]
NW_lbl = parc_to_NW.set_index("label").loc[parc_lbl, "network"].values

# Create dataframe for tests
disp_df = pd.DataFrame(np.vstack([vv, hemi, parc_lbl, NW_lbl, grad_SD.get_fdata()]).T, columns = ["Vtx", "Hemi", "Parc", "NW", "Disp_tot", "Disp_G1", "Disp_G2", "Disp_G3"])

clusters = nib.load(clusters_path).get_fdata().T
disp_df[["DispROI_tot", "DispROI_G1", "DispROI_G2", "DispROI_G3"]] = clusters

disp_df.to_csv(f"{output_dir}/{group}.gcca_dispersion.csv", index=False)
print(disp_df.head())

#---------------------------------------------------------------------------------------------------

# Compute individual average framewise displacement

radius = 50.0
rest_FD = []
for subj in subj_id:
    # Compute FD of traslations
    runs = ["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"]
    runwise_FD = 0
    for i, run in enumerate(runs):
        traslations = np.loadtxt(f"{subj_dir}/{subj}/MNINonLinear/Results/rfMRI_{run}/Movement_Regressors.txt",
                                       usecols=[0,1,2], dtype="float64")
        translation_deltas = np.abs(traslations).sum(axis=1)

        rotations = np.loadtxt(f"{subj_dir}/{subj}/MNINonLinear/Results/rfMRI_{run}/Movement_Regressors.txt",
                                       usecols=[3,4,5], dtype="float64")
        rotations_deltas = (abs(rotations) / 360) * (2 * np.pi * radius)
        rotations_deltas = rotations_deltas.sum(axis=1)

        FD = translation_deltas + rotations_deltas
        runwise_FD += FD.sum()

    rest_FD.append(np.mean(runwise_FD))

#---------------------------------------------------------------------------------------------------

# Generate table with subject data for analyses

# Load cognitive scores
cog_df = pd.read_csv(cog_path,
                    usecols = ['Subject', 'PicVocab_Unadj', 'ReadEng_Unadj',
                               'CardSort_Unadj', 'Flanker_Unadj', 'ProcSpeed_Unadj',
                               'VSPLOT_TC', 'PMAT24_A_CR', 'PicSeq_Unadj',
                               'ListSort_Unadj', 'IWRD_TOT', 'CogFluidComp_Unadj',
                               'CogCrystalComp_Unadj'],
                     index_col = 'Subject').loc[subj_id,:]

cog_df["FD"] = rest_FD

# Calculate weighted factors
cog_df['G'] = (cog_df[['PicVocab_Unadj', 'ReadEng_Unadj', 'CardSort_Unadj', 'Flanker_Unadj', 'ProcSpeed_Unadj',
                       'VSPLOT_TC', 'PMAT24_A_CR', 'PicSeq_Unadj', 'ListSort_Unadj', 'IWRD_TOT']]
               * [.624, .642, .364, .259, .232, .578, .626, .354, .451, .294]).mean(axis=1)


# Load covariates for correcting cognitive scores
covar_df = pd.merge(pd.read_csv(cog_path,
                                usecols=['Subject', 'Gender'],
                                dtype={'Subject':'int32', 'Gender':'category'}),
                    pd.read_csv(rest_path,
                                usecols=['Subject', 'Age_in_Yrs', 'Handedness', 'SSAGA_Educ'],
                                dtype={'Subject':'int32', 'Age_in_Yrs':'int32', 'Handedness':'int32'}),
                    ).set_index('Subject')

covar_df = covar_df.loc[subj_id]
cog_df = cog_df.merge(covar_df, left_index=True, right_index=True)


# Add aggregated gradient value of the ROIs
ROIs = pd.read_csv(f'{output_dir}/{group}.gcca_dispersion.csv')
cog_df.index = cog_df.index.astype('str')

grads = np.load(f'{gcca_dir}/{group}.gcca.32k_fs_LR.npy').copy()

for grd in ['tot', 'G1', 'G2', 'G3']:
    for i in range(3):
        grads_df = pd.DataFrame(np.vstack([ROIs[f"DispROI_{grd}"], grads[:, i, :]]).T, columns=np.hstack(["ROI", subj_id]))
        grads_avg = grads_df.groupby("ROI").agg(np.mean).reset_index().T.iloc[:,1:].drop('ROI')
        cols = [f"G{i+1}_ROI{int(n)}_Disp{grd}" for n in grads_avg.columns]
        grads_avg = grads_avg.rename(columns=dict(zip(grads_avg.columns, cols))).rename_axis('Subject')
        cog_df = cog_df.merge(grads_avg, left_index=True, right_index=True)



cog_df.to_csv(f'{output_dir}/{group}.cog_data.csv')

print(cog_df.head())

print("Correlation between intelligence measures:\n",
      cog_df[["CogCrystalComp_Unadj", "CogFluidComp_Unadj", "G"]].corr())

#----------------------------------------------------------------------------------------------------

# Measure clusters' overlap with networks

disp_df = pd.read_csv(f"{output_dir}/{group}.gcca_dispersion.csv", header=0, usecols=["Vtx", "NW", "DispROI_tot"])
disp_df = disp_df[disp_df["DispROI_tot"] > 0]
ROI_names = [f"ROI{int(i)+1}" for i in np.arange(disp_df["DispROI_tot"].max())]
print(ROI_names)

clusters = disp_df.groupby(["DispROI_tot", "NW"]).agg("count")["Vtx"].reset_index().pivot(index="DispROI_tot", columns="NW", values="Vtx")
clusters.columns = np.array(nw_name)[clusters.columns.astype("int32")-1]
clusters.loc["TOT", :] = clusters.sum()
clusters.loc[:, "TOT"] = clusters.sum(axis=1)

clusters.to_csv(f"{output_dir}/{group}.dispROIs_{nw_N}NWs_overlap.csv")


missing_NWs = np.asarray(nw_name)[~np.isin(nw_name, clusters.columns)]
missing_NWs = pd.DataFrame(np.full([len(clusters),  len(missing_NWs)], np.nan), columns=missing_NWs, index=clusters.index)
clusters = clusters.merge(missing_NWs, left_index=True, right_index=True)[np.hstack([nw_name, 'TOT'])]

clusters = (clusters.T / clusters.T.sum() * 100).T * 2

clusters.rename(columns={"SalVentAttn":"VAN", "DorsAttn":"DAN", "Default":"DMN", "Cont":"FPN"},
                index=dict(zip(clusters.index, ROI_names)), inplace=True)
clusters = clusters.loc[ROI_names, :"TOT"]
clusters[clusters.isna()] = 0

print(clusters)

#-------------------------------------------------------------------------------------------------
