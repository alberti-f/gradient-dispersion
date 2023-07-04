# Import dependencies

import os
import nibabel as nib
import numpy as np
from scipy import stats
import pandas as pd
import ciftools_FA as ct
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import subprocess as sp
from joblib import Parallel, delayed
from settings import root_dir, output_dir, subj_id, group, lbl_N, nw_N, networks, networks_txt, NW_tbl, nw_name, nj
subj_dir = f"{root_dir}/Subjects/"
gcca_dir = f"{output_dir}/GCCA"
#----------------------------------------------------------------------------------------------------



''' Define functions to parallelize permutation testing '''

def perm_fit(model, data, perm_cols):
    data.loc[:, perm_cols] = np.random.permutation(data.loc[:, perm_cols])
    lm_perm = ols(model, data=data).fit()
    return lm_perm.fvalue, lm_perm.tvalues

def perm_lm(model, data, perm_n, perm_cols=None, n_jobs=1):
    lm = ols(model, data=data).fit()

    A = np.identity(len(lm.params))
    A = A[1:,:]
    F_results = {'Endog': lm.model.endog_names, 'F': lm.fvalue, 'R2': lm.rsquared, 'R2_adj': lm.rsquared_adj}
    dfs = [lm.params, lm.tvalues, lm.bse]
    t_results = pd.DataFrame(np.vstack(dfs).T, index=lm.model.exog_names, columns=['Coef', 't', 'SE'])
    
    F_null = []
    t_null = []
    data_perm = data.copy()
    if perm_cols is None:
        perm_cols = data.columns[data.columns.isin(lm.model.exog_names)]
        
    r= Parallel(n_jobs=n_jobs)(delayed(perm_fit)(model, data_perm, perm_cols) for i in range(perm_n))
    F_null, t_null = zip(*r)
    
    t_p = [sum(np.abs(np.asarray(t_null).T[n]) > np.abs(t_results.t[n])) / perm_n for n in range(len(t_results))]
    t_results['p'] = t_p
    F_results['p'] = (sum(F_null > F_results['F']) / perm_n)

    return F_results, t_results, np.array(F_null), np.array(t_null)
#----------------------------------------------------------------------------------------------------



''' Run permutation tests '''

# Aggregate highly correlated predictors
cog_df = pd.read_csv(f'{output_dir}/{group}.cog_data.csv', index_col=0, header=0)
G = 'G1'
cog_df[f'{G}_ROI38_Disptot'] = np.mean(cog_df[[f'{G}_ROI3_Disptot', f'{G}_ROI7_Disptot']], axis=1)
cog_df[f'{G}_ROI15_Disptot'] = np.mean(cog_df[[f'{G}_ROI1_Disptot', f'{G}_ROI5_Disptot']], axis=1)
G = 'G2'
cog_df[f'{G}_ROI38_Disptot'] = np.mean(cog_df[[f'{G}_ROI3_Disptot', f'{G}_ROI7_Disptot']], axis=1)
cog_df[f'{G}_ROI15_Disptot'] = np.mean(cog_df[[f'{G}_ROI1_Disptot', f'{G}_ROI5_Disptot']], axis=1)
G = 'G3'
cog_df[f'{G}_ROI38_Disptot'] = np.mean(cog_df[[f'{G}_ROI3_Disptot', f'{G}_ROI7_Disptot']], axis=1)
cog_df[f'{G}_ROI15_Disptot'] = np.mean(cog_df[[f'{G}_ROI1_Disptot', f'{G}_ROI5_Disptot']], axis=1)


comp_cols = ['CogFluidComp_Unadj', 'CogCrystalComp_Unadj', 'G']
ROI_cols = np.array(['G1_ROI2_Disptot', 'G1_ROI4_Disptot', 'G1_ROI6_Disptot', 'G1_ROI7_Disptot', 'G1_ROI15_Disptot', 'G1_ROI38_Disptot'])
covars = ['C(Gender)', 'Age_in_Yrs', 'Handedness', 'SSAGA_Educ']


# Correct for covariates
X = ' + '.join(covars)
for col in comp_cols:
    lm = ols(f'{col} ~ {X}', data=cog_df).fit()
    cog_df[col] = stats.zscore(lm.resid, axis=0)


# Run permutation tests
perm_n = 10#000
X = ' + '.join(ROI_cols)
F_tests = []
F_test_p = []

results = {}
for y in comp_cols:
    F, t, F_null, t_null = perm_lm(f'{y} ~ {X}', cog_df, perm_n, perm_cols=None, n_jobs=-2)
    results[y] = {'F': F, 't': t, 'F_null': F_null, 't_null': t_null}
    
    
# Correct multiple comparisons
pF_adj = sm.stats.multipletests([values['F']['p'] for key, values in results.items()], method='fdr_bh', alpha=0.05)[1]

for i, key in enumerate(results.keys()):
    results[key]['F']['p_adj'] = pF_adj[i]
    results[key]['t']['p_adj'] = np.hstack(['',sm.stats.multipletests(results[key]['t']['p'][1:], method='fdr_bh', alpha=0.05)[1]])

global_results = pd.DataFrame([results['CogFluidComp_Unadj']['F'], results['CogCrystalComp_Unadj']['F'], results['G']['F']])

print(global_results)

for key, value in results.items():
    if results[key]['F']['p'] < 0.05:
        print(f"post-hoc:\t{key}\n", pd.DataFrame(results[key]['t']), '\n')
        
#----------------------------------------------------------------------------------------------------
        
        
        
''' Test correlation with individual tests '''        
        
# test correlation between Fluid intelligence and its components
r, p = stats.spearmanr(cog_df[['CardSort_Unadj', 'Flanker_Unadj', 'PMAT24_A_CR', 'PicSeq_Unadj', 'ListSort_Unadj', 'CogFluidComp_Unadj']], nan_policy='omit')

cols = ['CardSort_Unadj', 'Flanker_Unadj', 'PMAT24_A_CR', 'PicSeq_Unadj', 'ListSort_Unadj', 'CogFluidComp_Unadj', 'G1_ROI6_Disptot']
r, p = stats.spearmanr(cog_df[cols], nan_policy='omit')
r = pd.DataFrame(r, index=cols, columns=cols)
p = pd.DataFrame(p, index=cols, columns=cols)

h, pAdj, _, _ = sm.stats.multipletests(p.loc['G1_ROI6_Disptot', 'CardSort_Unadj':'ListSort_Unadj'], method='fdr_bh')

results = pd.DataFrame(np.vstack([r.loc['G1_ROI6_Disptot', 'CardSort_Unadj':'ListSort_Unadj'], pAdj, h]).T,
             columns=['rho', 'p_adj', 'h'], index=['CardSort_Unadj', 'Flanker_Unadj', 'PMAT24_A_CR', 'PicSeq_Unadj', 'ListSort_Unadj'])

print(results)
