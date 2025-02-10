#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:41:46 2025

@author: alexgorelick
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import plotnine
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import itertools
import seaborn as sns
import os

from scipy.stats import pearsonr, spearmanr
from plotnine import ggplot, geom_histogram, aes, labs, geom_bar, geom_hline, geom_text, facet_wrap, geom_point, geom_smooth, geom_boxplot, position_jitter, geom_text
from collections import Counter 
from joblib import Parallel, delayed
from missforest import MissForest # pip install MissForest lightgbm tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader 

# set working directory
os.chdir('/home/alex/repos/website_projects/cell_line_drug_sensitivity')

# use CUDA (GPU) if available otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# load/merge raw data
# https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html
# =============================================================================

# load the drug response data for the two datasets
drugdata1 = pd.read_excel('original_data/GDSC1_fitted_dose_response_27Oct23.xlsx')
drugdata2 = pd.read_excel('original_data/GDSC2_fitted_dose_response_27Oct23.xlsx')

# extra the drug metadata
drug_info1 = drugdata1.drop_duplicates(subset=['DRUG_ID']).filter(['DRUG_ID','DRUG_NAME','PUTATIVE_TARGET','PATHWAY_NAME','COMPANY_ID','MIN_CONC','MAX_CONC'])
drug_info1['dataset'] = 'GDSC1'
drug_info2 = drugdata2.drop_duplicates(subset=['DRUG_ID']).filter(['DRUG_ID','DRUG_NAME','PUTATIVE_TARGET','PATHWAY_NAME','COMPANY_ID','MIN_CONC','MAX_CONC'])
drug_info2['dataset'] = 'GDSC2'
drug_info = pd.concat([drug_info1, drug_info2], axis=0)

# remove drugs missing names
drug_info = drug_info[pd.isna(drug_info.DRUG_NAME)==False]
drug_info['DRUG_ID'] = drug_info['DRUG_ID'].astype(int)
drug_info = drug_info.sort_values('DRUG_ID')

# pivot to wide tables for each dataset
auc1 = drugdata1.pivot(index='COSMIC_ID', columns='DRUG_ID', values='AUC')
auc2 = drugdata2.pivot(index='COSMIC_ID', columns='DRUG_ID', values='AUC')

# use the cell-lines with data in both datasets (to have the maximum number of drugs tested)
auc1_lines = set(auc1.index)
auc2_lines = set(auc2.index)
lines_in_both = list(set.intersection(auc1_lines, auc2_lines)) # 76 drugs in both datasets
auc1 = auc1.loc[lines_in_both]
auc2 = auc2.loc[lines_in_both]

# check how the two datasets line up. 
auc1_drugs = set(auc1.columns)
auc2_drugs = set(auc2.columns)
drugs_in_both = list(set.intersection(auc1_drugs, auc2_drugs)) 
len(drugs_in_both) # 76 drugs in both datasets

# to make life easy, use second dataset for drugs present in both (maybe there was a reason to re-screen the drug) 
drugs_only_in_dataset1 = list(auc1_drugs.difference(drugs_in_both))
auc = pd.concat([auc1.filter(drugs_only_in_dataset1), auc2], axis=1)
print(f' # cell lines: {auc.shape[0]}, # drugs: {auc.shape[1]}')


# =============================================================================
# make a simple 3-class classification for sensitive/intermediate/resistant
# based on AUC percentiles
# =============================================================================

auc3 = auc.copy()
n_lines = auc.shape[0]
drugs = auc.columns.to_list()
for d in drugs:
    auc3[d] = 1
    q33, q66 = auc[d].quantile([0.333,0.666])
    auc3.loc[(auc[d] < q33),d] = 0
    auc3.loc[(auc[d] > q66),d] = 2
    auc3.loc[np.isnan(auc[d]),d] = np.nan # preserve missing values

auc = auc3.copy()
del(auc3) # cleanup


# =============================================================================
# load in the expression data and make sure the cell lines align with the AUC data
# =============================================================================

# load preprocessed gene expression data
expr = pd.read_csv('original_data/Cell_line_RMA_proc_basalExp.txt', delimiter='\t')
expr.index = expr.GENE_SYMBOLS
expr = expr.drop(['GENE_title','GENE_SYMBOLS'], axis=1)
expr = expr.transpose()
expr['COSMIC_ID'] = expr.index.str.replace('DATA.','')
expr = expr.set_index('COSMIC_ID')

# line up the cell lines (rows for the expression auc the AUC auc)
auc.index = auc.index.astype(str)
expr_lines = set(expr.index)
auc_lines = set(auc.index)
common_lines = list(set.intersection(expr_lines, auc_lines))
expr = expr.loc[common_lines]
auc = auc.loc[common_lines]

# remove duplicate/nan gene names
cleaned_columns = expr.columns.unique()
cleaned_columns = cleaned_columns[pd.isna(cleaned_columns)==False]
expr = expr.loc[:,expr.columns.isin(cleaned_columns)]


# =============================================================================
# remove drugs missing AUC values in too many cell lines
# =============================================================================

# check fraction of genes with NA expression in any cell line
np.mean(np.isnan(expr).apply(sum, 0) > 0)

# check fraction of drugs with NA AUCs in any cell line
np.mean(np.isnan(auc).apply(sum, 0) > 0)

nans_per_drug = np.isnan(auc).apply(sum, 0)
tmp = pd.DataFrame({'drug': nans_per_drug.index.astype(str), 'n_lines': nans_per_drug})
tmp = tmp.sort_values('n_lines', ascending=True)
tmp['pct_of_lines'] = 100*tmp['n_lines'] / auc.shape[0]
tmp['drug'] = pd.Categorical(tmp['drug'], categories=pd.unique(tmp['drug']))
tmp['pos'] = range(tmp.shape[0])

# plot the number of cell lines with missing AUCs per drug
p = (
     ggplot(data=tmp) 
     + geom_bar(aes(x='pos', y='pct_of_lines'), color='steelblue', size=0.5, stat='identity')
     + geom_hline(yintercept=10, linetype='dashed', color='red', size=0.75) 
     + geom_text(x=25, y=9.2, label='10%', color='red') 
     + labs(x='All drugs', y='% of cell lines', title='% of cell lines with missing AUC per drug')
     )

# let's only consider the drugs that have are missing AUC data for at most 10% of cell lines. We can then impute missing values.
tmp = tmp[tmp.pct_of_lines <= 10]
tmp.loc[:,'Drug_ID'] = tmp['drug'].astype(int)
auc = auc.filter(tmp.Drug_ID)


# =============================================================================
# split data into training/testing before any more processing
# =============================================================================

# randomly split train/test data
X_train, X_test, Y_train, Y_test = train_test_split(expr, auc, test_size = 0.2, shuffle=True, random_state=123)

# check the dimensions of the training/testing input data and targets
print(f'Expr data (training):\t# cell lines: {X_train.shape[0]}, # drugs: {X_train.shape[1]}')
print(f'AUC data (training):\t# cell lines: {Y_train.shape[0]}, # drugs: {Y_train.shape[1]}')
print(f'Expr data (testing):\t# cell lines: {X_test.shape[0]}, # drugs: {X_test.shape[1]}')
print(f'AUC data (testing):\t# cell lines: {Y_test.shape[0]}, # drugs: {Y_test.shape[1]}')



# =============================================================================
# feature selection
# we have 200x more features than samples (rows) in our training data, not good.
# let's prioritize a smaller subset of features
# =============================================================================

# first lets choose a narrower class of drugs (which still has numerous drugs)
counts = Counter(drug_info['PATHWAY_NAME'])

# subset data for specific drugs
drug_info_subset = drug_info[drug_info['PATHWAY_NAME']=='ERK MAPK signaling']
drug_info_subset = drug_info[drug_info['PATHWAY_NAME']=='ERK MAPK signaling']
valid_drugs = list(set.intersection(set(drug_info_subset.DRUG_ID), set(auc.columns)))

stat_list = []
pval_list = []
gene_list = []
drug_list = []
drugs = valid_drugs
drugs.sort()
genes = X_train.columns.to_list()
genes.sort()

def compute_ranksum(g):
    return g, stats.ranksums(X_train.loc[Y_train[d] == 0, g], X_train.loc[Y_train[d] == 2, g])

# this will take a while, grab a coffee
for i, d in enumerate(drugs):
    print(f'This drug: {d}')
    if i % 10 == 0: print(f'{i}/{len(drugs)} drugs tested.')
    
    # use Parallel and delayed functions so that we use all available cores for parallel processing
    results = Parallel(n_jobs=-1)(delayed(compute_ranksum)(g) for g in genes)

    # extend lists for our output
    drug_list.extend([d] * len(genes))
    gene_list.extend([r[0] for r in results])
    stat_list.extend([r[1][0] for r in results])
    pval_list.extend([r[1][1] for r in results])

res = pd.DataFrame({'drug': drug_list, 'gene': gene_list,'stat': stat_list, 'pval': pval_list})
#res.to_csv('processed_data/feature_selection_test_results.tsv', sep='\t')
#res = pd.read_csv('processed_data/feature_selection_test_results.tsv', delimiter='\t')
pval_data = res.pivot(index='gene',columns='drug',values='pval')
stat_data = res.pivot(index='gene',columns='drug',values='stat')

# subset for genes which had significant effects on sensitivity to at least 50% of drugs
alpha = 0.05 / pval_data.shape[1]
sig = (pval_data < alpha) #& (ests < 0)
myres = pd.DataFrame({'gene': pval_data.index, 'frac_sig': sig.apply(np.mean, 1)})
myres = myres.sort_values('frac_sig', ascending=False)
myres = myres[myres['frac_sig'] > 0.8] 
gene_list = myres['gene'].tolist()
gene_list.sort()
len(gene_list)

# subset the training/testing expression data for the subset genes
valid_gene_list = set.intersection(set(gene_list), set(X_train.columns))
valid_gene_list = list(valid_gene_list)
valid_gene_list.sort()
X_train_reduced = X_train.filter(valid_gene_list)
X_test_reduced = X_test.filter(valid_gene_list)
X_train_reduced.index = X_train.index
X_test_reduced.index = X_test.index

# subset the training/testing AUC data for the subset drugs
Y_train_reduced = Y_train.filter(valid_drugs)
Y_test_reduced = Y_test.filter(valid_drugs)

# check the dimensions of the training/testing input data and targets
print(f'Expr data (training):\t# cell lines: {X_train_reduced.shape[0]}, # genes: {X_train_reduced.shape[1]}')
print(f'AUC data (training):\t# cell lines: {Y_train_reduced.shape[0]}, # drugs: {Y_train_reduced.shape[1]}')
print(f'Expr data (testing):\t# cell lines: {X_test_reduced.shape[0]}, # genes: {X_test_reduced.shape[1]}')
print(f'AUC data (testing):\t# cell lines: {Y_test_reduced.shape[0]}, # drugs: {Y_test_reduced.shape[1]}')



#%%
# =============================================================================
# impute missing values in the *training* target matrix based on random forest
# we do this after subsetting for the shortlist of drugs because we don't want to 
# include AUCs from very different drugs in the imputation
# =============================================================================

imputer = MissForest()
Y_train_reduced_imputed = imputer.fit_transform(Y_train_reduced)
Y_train_reduced_imputed = round(Y_train_reduced_imputed)

# correct any imputed values outside 0-1 after RF
Y_train_reduced_imputed = Y_train_reduced_imputed.clip(lower=0, upper=2)
Y_train_reduced_imputed = Y_train_reduced_imputed.filter(Y_train_reduced.columns)


#%% Finally!


# convert the train/test data to tensors
X_train_reduced_torch = torch.as_tensor(np.array(X_train_reduced), dtype=torch.float32, device=device)
X_test_reduced_torch = torch.as_tensor(np.array(X_test_reduced), dtype=torch.float32, device=device)
Y_train_reduced_imputed_torch = torch.as_tensor(np.array(Y_train_reduced_imputed), dtype=torch.float32, device=device)
Y_test_reduced_torch = torch.as_tensor(np.array(Y_train_reduced), dtype=torch.float32, device=device)

class RegressionDataSet(Dataset):
    def __init__(self, X, y):
        self.X = X.to(device)
        self.y = y.to(device)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(dataset = RegressionDataSet(X_train_reduced_torch, Y_train_reduced_imputed_torch), batch_size=8, shuffle=True)
test_loader = DataLoader(dataset = RegressionDataSet(X_test_reduced_torch, Y_test_reduced_torch), batch_size=8, shuffle=True)



# %% model
# set up model class
# topology: fc1, relu, fc2
# final activation function??

class MultiClassMultiLabelNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size*3)
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x) 
        #x = self.relu(x)
        x = self.fc2(x) # output raw logits
        x = x.view(-1, output_size, 3)  # Reshape to (batch_size, outputsize, 3)
        #x = self.softmax(x)
        x = self.log_softmax(x)
        return x
    
# define input and output dim
input_size = X_train_reduced.shape[1]; input_size
output_size = Y_train_reduced_imputed.shape[1]; output_size
hidden_size = 80

# %% training loop

# create a model instance
model = MultiClassMultiLabelNet(input_size=input_size, output_size=output_size, hidden_size=hidden_size).to(device)
loss_fn = nn.CrossEntropyLoss()

# optimizer
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
#optimizer=torch.optim.SGD(model.parameters(), lr=lr)
losses = []
number_epochs = 1000

# implement training loop
for epoch in range(number_epochs):
    for j, (X, y) in enumerate(train_loader): # enumerate gives us the list X, y
        
        y = y.to(device).to(torch.long) # why
        
        # optimization
        optimizer.zero_grad()

        # forward pass
        y_pred = model(X)

        # compute loss
        #loss = loss_fn(predictions, y)
        loss = loss_fn(y_pred.permute(0, 2, 1), y)  # Permute to (batch_size, 3, n_targets)

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

    current_loss = float(loss.data.detach().cpu().numpy())
    losses.append(current_loss)
        
    # print epoch and loss
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, loss: {loss.data.item()}")
    
sns.lineplot(x= range(len(losses)), y = losses)
    

#%% make predictions on the test data

with torch.no_grad():
    y_pred = model(X_test_reduced_torch.to(device))  # Output shape: (batch_size, 100, 3)
    y_pred_classes = torch.argmax(y_pred, dim=-1)  # Get class indices (batch_size, 100)
    y_pred_classes = y_pred_classes.cpu().numpy()

    # get the prob of each class
    probabilities = F.softmax(y_pred, dim=-1).cpu().numpy()

pred = pd.DataFrame(y_pred_classes)
pred.index = Y_test_reduced.index
pred.columns = Y_test_reduced.columns


#%% get the accuracy for each cell line


accuracies = []
for d in valid_drugs:
    y_pred = pred[d]
    y_obs = Y_test_reduced[d]
    available = np.isnan(y_obs)==False
    y_pred = y_pred[available]
    y_obs = y_obs[available].astype(int)
    acc = accuracy_score(y_pred, y_obs)
    accuracies.append(acc)

result = pd.DataFrame({'drug': valid_drugs, 'accuracy': accuracies})
result = result.sort_values('accuracy',ascending=False)
metadata = drug_info.drop_duplicates(subset=['DRUG_ID']).filter(['DRUG_ID','DRUG_NAME','PUTATIVE_TARGET','PATHWAY_NAME','COMPANY_ID'])
result = pd.merge(result, metadata, how='left', left_on='drug', right_on='DRUG_ID')


# what can we say about the predicted most-sensitive and most-resistant cell lines to Refametinib?
tmp = pred[1526]
pred_sens = tmp[tmp==0].index.tolist()
pred_res = tmp[tmp==2].index.tolist()

expr_pred_sens = X_test_reduced.loc[pred_sens]
expr_pred_res = X_test_reduced.loc[pred_res]
expr_pred = pd.concat([expr_pred_sens, expr_pred_res], axis=0)
expr_pred = expr_pred.transpose()

sns.set(font_scale=0.5) 
col_colors = pd.DataFrame({'Sensitive': ['b']*expr_pred_sens.shape[0] + ['r']*expr_pred_res.shape[0]})
col_colors.index = expr_pred.columns
g = sns.clustermap(expr_pred, col_colors=col_colors, dendrogram_ratio=0.15, figsize=[8,8], yticklabels = 1, xticklabels=1)
g.figure

# which cells are predicted resistant correctly?
drug_of_interest = pred.columns.tolist().index(1526)
prob_res = []
prob_sens = []
for l in range(pred.shape[0]):
    prob_sens.append(probabilities[l][drug_of_interest][0])
    prob_res.append(probabilities[l][drug_of_interest][2])

my_probs = pd.DataFrame({'COSMIC_ID': pred.index, 'prob_sens': prob_sens, 'prob_res': prob_res, 'true_class': Y_test_reduced[1526]})
my_probs.loc[my_probs['true_class']==0,'Class'] = 'Sensitive'
my_probs.loc[my_probs['true_class']==2,'Class'] = 'Resistant'
my_probs = my_probs[my_probs['Class'].isin(['Sensitive','Resistant'])]


drug_name = drug_info[drug_info['DRUG_ID']==1526]['DRUG_NAME'].tolist()[0]

p1 = (
     ggplot(data=my_probs)   
     + geom_point(aes(x='Class', y='prob_sens'), color='steelblue', size=1.5, position=position_jitter(width=0.15, height=0)) 
     + geom_boxplot(aes(x='Class', y='prob_sens'), color='black', width=0.2, fill=None, outlier_shape='')
     + labs(x='True sensitivity to drug', y='Probability', title=str('Probability of cell line sensitivity to ')+drug_name)
     )

p2 = (
     ggplot(data=my_probs)   
     + geom_point(aes(x='Class', y='prob_res'), color='steelblue', size=1.5, position=position_jitter(width=0.15, height=0)) 
     + geom_boxplot(aes(x='Class', y='prob_res'), color='black', width=0.2, fill=None, outlier_shape='')
     + labs(x='True sensitivity to drug', y='Probability', title=str('Probability of cell line resitance to ')+drug_name)
)







# the top cluster looks especially predictive, let's see what MSigDB says about it
clustered_genes = g.data2d.index.tolist()
hit = clustered_genes.index('TXNDC16')
cluster1 = clustered_genes[0:hit+1]
cluster2 = clustered_genes[hit+2:]
paste_cluster1 = " ".join(cluster1)
paste_cluster2 = " ".join(cluster2)


cluster1 = []
stop = False
for g in clustered_genes:
    if(g=='TXNDC16'):
        stop()
    else:
        cluster1.append(g)


g = g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 6)






#%% get the accuracy for each drug






with torch.no_grad():
    Y_test_pred = pd.DataFrame(model(X_test_reduced_torch).cpu().numpy())
    
Y_test_pred.index = Y_test.index
Y_test_pred.columns = Y_test.columns


# plot obs vs predicted AUCs for each drug
obs = Y_test.melt(ignore_index=False)
obs.columns = ['Drug_ID','AUC_obs']
pred = Y_test_pred.melt(ignore_index=False)
pred.columns = ['Drug_ID','AUC_pred']
merged = pd.merge(obs, pred, how='inner', on=['COSMIC_ID','Drug_ID'])
merged['COSMIC_ID2'] = merged.index
#merged = merged.reset_index
#merged.index = range(merged.shape[0])

p = (
     ggplot(data=merged) 
     + geom_point(aes(x='AUC_obs', y='AUC_pred'), color='steelblue', size=0.25)
     + facet_wrap(facets="~Drug_ID")
     + geom_smooth(aes(x='AUC_obs', y='AUC_pred'), method='lm')
     + labs(x='Observed AUC', y='Predicted AUC', title='', subtitle='RTK pathway drugs')
     )
p

     #+ facet_wrap(facets="~Drug_ID")

Y_test.pivot()
pd.DataFrame(Y_test_pred)
    
    