#Imports

import sys

import pandas as pd

import scipy
from scipy import stats

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

import xgboost as xgb
from xgboost import plot_importance

#outside scripts
from data_scripts import dataclean
from data_scripts import datapull
import gen_results_gb

#quandl pull

dataset = datapull.quandl_pull('/Users/stefankaehler/seattle_galv/recession_predictor/data/datasources2.csv')

#yieldcurve pull

yields = datapull.yieldcurve_pull('/Users/stefankaehler/seattle_galv/recession_predictor/data/Fed10Y_3M.csv')

#converting GDP quarterly data into monthly - need to convert it so it fills in the following 3 months

dataset = dataclean.convert_q_to_m(dataset, 'GDP')

#converting consumer sentiment into monthly

dataset = dataclean.convert_q_to_m(dataset, 'CONS_SENT')

#offsetting dataset to simulate data release timing

dataset = dataclean.timing_offset(dataset)

#calculating change in GDP and converting Y into categorical values 
dataset['Recession'] = ((dataset['GDP'] - dataset['GDP'].shift(3)) < 0).astype(int)
dataset = dataset.drop(columns = ['GDP']) #dropping calc column and recession column from dataset, experimenting with taking out fed funds rate

#merge fed interest rate data here
dataset = dataset.join(yields, how='outer')

#refining dataframe period to be 1959 to present. NOTE: this will need to be adjusted when run in the future

dataset = dataset[552:]
dataset = dataset[:-11]

#substituting mean value in for missing values and adding dummy column to indicate where done.

for col in dataset.columns:
    if str(col)=='Recession':
        continue
    dataclean.clean_zeros(col, dataset)
    
dataset['3YRT'] = dataset['3YRT'].fillna(dataset['3YRT'].mean())

#adding momentum factors

momentum_cols = list(dataset.columns[:-6])

momentum_cols.remove('PPI') #removing PPI and CPI because they need a different transformation
momentum_cols.remove('CPI')
momentum_cols.remove('Recession')

for i in [1,3,12]:
    for col in momentum_cols:
        if 'PXY' in str(col): #adding logic so it doesn't create a momentum column out of PXY columns
            continue
        dataclean.create_momentum(col,dataset,i)
        
#CPI Calcs

for i in [1,3,12]:
    for col in ['CPI','PPI']:
        dataclean.infl_momentum(col,dataset,i)
        
#split off X and y from base dataframe

y = dataset['Recession']
dataset = dataset.drop(columns = ['Recession'])
X = dataset

### Data Prep Finished Here ###

#generating probability table for test set

time = [0,-1,-3,-12]

params = {0:[4,0.005,500],-1:[5,0.005,500],-3:[10,0.01,300],-12:[1,0.001,1500]}

probs_chart = gen_results_gb.create_probs(X,y,12,723,550,time, params)[0]

#generating chart

gen_results_gb.create_chart(probs_chart,0)

performance = gen_results_gb.create_probs(X,y,12,723,550,time, params)[1]

print(performance)

gen_results_gb.create_feat_imp(X,y,12,723,550,-1, params)

print(gen_results_gb.create_probs(X,y,12,723,550,time)[0].tail(5))