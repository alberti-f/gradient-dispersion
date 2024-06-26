# Import dependencies

from settings import *
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from joblib import Parallel, delayed
from itertools import combinations
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore, spearmanr
from statsmodels.stats.multitest import multipletests
import sys, os, re

#----------------------------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


''' Define functions to parallelize permutation testing '''

def t_value(regressor, X, y):
    y_pred = regressor.predict(X)
    mse = mean_squared_error(y, y_pred)
    std_error = np.sqrt(mse)
    t = regressor.coef_ / std_error
    return t


def f_value(regressor, X, y):
    
    y_pred = regressor.predict(X)
    msm = sum((y_pred - y.mean())**2) / (X.shape[1] - 1)
    mse = mean_squared_error(y, y_pred)
    f = msm / mse

    return f


def single_cv(train_index, test_index, data, X_cols, y_col, n_perm, nj):
    train_index, test_index = data.index[train_index], data.index[test_index]
    X_train, y_train = data.loc[train_index, X_cols], data.loc[train_index, y_col]
    X_test, y_test = data.loc[test_index, X_cols], data.loc[test_index, y_col]

    X_train = zscore(X_train, axis=0)
    X_test = zscore(X_test, axis=0)
    y_train = zscore(y_train)
    y_test = zscore(y_test)

    lm = LinearRegression(fit_intercept=False).fit(X_train, y_train)
    nlm = LinearRegression(fit_intercept=False)

    null_lm = Parallel(n_jobs=nj, prefer="processes")(delayed(nlm.fit)(X_train, shuffle(y_train)) for _ in range(n_perm))
    null_f = Parallel(n_jobs=nj, prefer="processes")(delayed(f_value)(nlm, X_test, shuffle(y_test)) for nlm in null_lm)
    null_t = Parallel(n_jobs=nj, prefer="processes")(delayed(t_value)(nlm, X_test, shuffle(y_test)) for nlm in null_lm)
    null_t = np.asanyarray(null_t)
    
    f = f_value(lm, X_test, y_test)
    t = t_value(lm, X_test, y_test)

    f_p = sum(null_f > f) / n_perm
    t_p = (np.abs(null_t) > np.abs(t)).sum(axis=0) / n_perm
    t_p_adj = multipletests(t_p, method='fdr_bh')[1]

    y_pred = lm.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r = np.corrcoef(y_test, y_pred)[0,1]

    model_results = {'F': f, 'F_p': f_p, 'R2': r2, 'MSE': mse, 'R': r}
    phoc_results = {'coeff': lm.coef_, 't': t, 't_p': t_p, 't_p_adj': t_p_adj}

    return model_results, phoc_results, null_f, null_t


def perm_lm_cv(splits, data, X_cols, y_col, n_perm, nj):

    f_results = {}
    t_results = {}
    f_null = []
    t_null = []

    for i, (train_index, test_index) in enumerate(splits):
        f, t, null_f, null_t = single_cv(train_index, test_index, data, X_cols, y_col, n_perm, nj)
        t = pd.DataFrame(t, index=X_cols)
        t["fold"] = i+1

        f_results[f'k{i+1}'] = f
        t_results[f'k{i+1}'] = t
        f_null.append(null_f)
        t_null.append(null_t)


    f_null = np.hstack(f_null)
    f_results = pd.DataFrame(f_results).T

    t_null = np.vstack(t_null).T
    t_results = pd.concat([val for _, val in t_results.items()])

    return f_results, t_results, f_null, t_null

#----------------------------------------------------------------------------------------------------


# Aggregate highly correlated predictors
disp_ROIs = pd.read_csv(f'{output_dir}/{group}.gcca_dispersion.csv', header=0, usecols=["DispROI_tot"])
disp_ROIs = np.int32(np.unique(disp_ROIs)[1:])
cog_df = pd.read_csv(f'{output_dir}/{group}.cog_data.csv', index_col=0, header=0).dropna()


G = "G1"
redundant = []
non_redundant = list(disp_ROIs)

for i, j in combinations(disp_ROIs, 2):
    if i in redundant:
        continue

    r = cog_df[[f'{G}_ROI{i}_Disptot', f'{G}_ROI{j}_Disptot']].corr().loc[f'{G}_ROI{i}_Disptot', f'{G}_ROI{j}_Disptot']

    if r >= 0.5 and np.int32(f"{i}{j}") not in non_redundant:
        cog_df[f'{G}_ROI{i}{j}_Disptot'] = cog_df[[f'{G}_ROI{i}_Disptot', f'{G}_ROI{j}_Disptot']].mean(axis=1)
        non_redundant.append(np.int32(f"{i}{j}"))
        non_redundant.remove(i)
        non_redundant.remove(j)
        redundant.extend([i,j])


# Correct for covariates
comp_cols = ['CogFluidComp_Unadj', 'CogCrystalComp_Unadj', 'G']
ROI_cols = np.array([f'G1_ROI{i}_Disptot' for i in non_redundant])
covars = ['Gender', 'Age_in_Yrs', 'Handedness', 'SSAGA_Educ', "FD"]


# Correct for covariates
covars = ['C(Gender)', 'Age_in_Yrs', 'Handedness', 'SSAGA_Educ', "FD"]
X = ' + '.join(covars)
for col in comp_cols:
    lm = ols(f'{col} ~ {X}', data=cog_df).fit()
    lm = lm.get_influence() 

    cog_df[col] = lm.resid_studentized_internal

#----------------------------------------------------------------------------------------------------

''' Run permutation tests '''

n_perm = 1000

# cog_df.dropna(inplace=True)
subj_id = cog_df.index
n_splits = 100
stratify_by = "Gender"


# If path to t-test results is provided runs validation test in hold-out set
# Otherwise, run cross validation on the training set
if len(sys.argv) == 1:
    print("\n\nRunning cross-validation on training set")
    print("Number of subjects:", len(cog_df))
    print("Number of permutations:", n_perm)
    print("Number of splits:", n_splits)
    print("Stratified by:", stratify_by)

    splits = list(StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=0).split(cog_df, cog_df[stratify_by]))

    results = {}
    for comp in comp_cols:
        results[comp] = {}
        results[comp]["F"], results[comp]['t'], null_f, null_t = perm_lm_cv(splits, cog_df, ROI_cols, comp, n_perm, nj)

    significant_models = []
    for comp in comp_cols:
        f_results = results[comp]["F"].mean()
        f_results["F_p"] = sum(null_f > f_results['F']) / len(null_f)
        print("\n\n", comp, ":")
        print(f_results)
        print("\n")

        weights = results[comp]["F"]["F"].values
        weights -= weights.min()
        t_results = results[comp]["t"].groupby(results[comp]["t"].index).agg('mean')
        t_results["coeff"] = results[comp]["t"]["coeff"].groupby(results[comp]["t"].index).agg(
                np.average, weights=weights)#kwargs={'weights': results[comp]["t"]["fold"]}
        t_avg = t_results.t.values.reshape(-1,1)
        t_results["t_p"] = (np.abs(null_t) > np.abs(t_avg)).sum(axis=1) / null_t.shape[1]
        t_results["t_p_adj"] = multipletests(t_results.t_p, method='fdr_bh')[1]
        print("\n", null_t.shape, t_avg.shape, t_results.shape, "\n")
        print(t_results)
        print("\n")

        

        t_results.to_csv(f'{output_dir}/{group}.{comp}_t_results.csv')

        if f_results["F_p"] < 0.05:
            significant_models.append(comp)




else:
    n_perm = n_perm * n_splits
    print("\n\nValidating model on hold-out set")
    print("Number of subjects:", len(cog_df))
    print("Number of permutations:", n_perm)

    train_dir = sys.argv[1]
    for comp in comp_cols:
        print(f"\n\n{comp}\n")

        # Load t-test results
        pattern = fr".*{comp}_t_results.*\.csv"
        for filename in os.listdir(train_dir):
            if re.match(pattern, filename):
                t_results = pd.read_csv(f"{train_dir}/{filename}", index_col=0).loc[ROI_cols]
                print("Train set model:\n", t_results.coeff)

        # Get model F and t values using the training set's coefficients
        X = zscore(cog_df[ROI_cols], axis=0)
        y = zscore(cog_df[comp])
        coeff_avg = t_results.coeff.values
        lm = LinearRegression(fit_intercept=False).fit(X, y)
        lm.coef_ = coeff_avg

        y_pred = lm.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        r = np.corrcoef(y, y_pred)[0,1]
        t = t_value(lm, X, y)
        F = f_value(lm, X, y)

        # Permutation test
        nlm = LinearRegression(fit_intercept=False)
        null_lm = Parallel(n_jobs=nj, prefer="processes")(delayed(nlm.fit)(X, shuffle(y)) for _ in range(n_perm))
        F_null = Parallel(n_jobs=nj, prefer="processes")(delayed(f_value)(nlm, X, shuffle(y)) for nlm in null_lm)
        t_null = Parallel(n_jobs=nj, prefer="processes")(delayed(t_value)(nlm, X, shuffle(y)) for nlm in null_lm)
        t_null = np.asanyarray(t_null)

        F_p = sum(F_null > F) / n_perm
        t_p = (np.abs(t_null) > np.abs(t)).sum(axis=0) / n_perm
        tp_adj = multipletests(t_p, method='fdr_bh')[1]

        t_results =  pd.DataFrame(np.vstack([lm.coef_, t, t_p, tp_adj]).T, index=ROI_cols, columns=['coeff', 't', 'p', 'p_adj'])

        print(f"F-test:\nF: {F}, R2: {r2}, MSE: {mse}, F_p: {F_p}")
        print(t_results)


#----------------------------------------------------------------------------------------------------


''' Test correlation with individual tests '''        
        
# test correlation between Fluid intelligence and its components
r, p = spearmanr(cog_df[['CardSort_Unadj', 'Flanker_Unadj', 'PMAT24_A_CR', 'PicSeq_Unadj', 'ListSort_Unadj', 'CogFluidComp_Unadj']], nan_policy='omit')

cols = ['CardSort_Unadj', 'Flanker_Unadj', 'PMAT24_A_CR', 'PicSeq_Unadj', 'ListSort_Unadj', 'CogFluidComp_Unadj', 'G1_ROI6_Disptot']
r, p = spearmanr(cog_df[cols], nan_policy='omit')
r = pd.DataFrame(r, index=cols, columns=cols)
p = pd.DataFrame(p, index=cols, columns=cols)

h, pAdj, _, _ = multipletests(p.loc['G1_ROI6_Disptot', 'CardSort_Unadj':'ListSort_Unadj'], method='fdr_bh')

corr_results = pd.DataFrame(np.vstack([r.loc['G1_ROI6_Disptot', 'CardSort_Unadj':'ListSort_Unadj'], pAdj, h]).T,
             columns=['rho', 'p_adj', 'h'], index=['CardSort_Unadj', 'Flanker_Unadj', 'PMAT24_A_CR', 'PicSeq_Unadj', 'ListSort_Unadj'])


print("Spearman correlation between individual tests and G1_ROI6_Disptot\n", corr_results)

#--------------------------------------------------------------------------------------------------