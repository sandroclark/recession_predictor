#Contains helper functions for cleaning dataset

import pandas as pd
import quandl
from scipy import stats
import scipy
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

def convert_q_to_m(df, col_name):
    
    i = 0
    
    for val in df[str(col_name)]:
        if val == 0:
            df[str(col_name)].iloc[i] = df[str(col_name)].iloc[i-1]
            #print(i)
    
        i += 1
    
    return df


def clean_zeros(col_name,df):
    if df[str(col_name)][df[str(col_name)]==0].count():
        df[str(col_name)+'_PXY'] = (df[col_name] == 0).astype(int)
        df[col_name] = df[col_name].mask(df[col_name] == 0,df[col_name].mean())
    return df

def create_momentum(column_name, df, shift_size):
    #this function creates a new column in your dataframe that is a momentum feature of another column
    df[str(column_name)+'_'+str(shift_size)+'m_shift'] = df[str(column_name)] - df[str(column_name)].shift(shift_size)
    return df

#CPI Calcs

def infl_momentum(column_name, df, shift_size):
    #this function creates a new column in your dataframe that is a momentum feature of another column
    df[str(column_name)+'_'+str(shift_size)+'m_shift'] = ((df[str(column_name)] - df[str(column_name)].shift(shift_size))/df[str(column_name)].shift(shift_size))*100

    
#offseting data to factor in release timing
def timing_offset(df):
    offset_dict = {'PMI':1, 'UNR':1, 'YUNR':1,'US_NHOME_SALES':1, 'PART_TIME':1, 'CPPR_PRICE':1, 'HOUS_PERMS':1, 'HOUS_STARTS':1, 'CAP_UTIL':1, 'PERS_SAVINGS':3, 'EXPORTS':3, 'IMPORTS':3, 'TRADE_BALANCE':3, 'US_M2':1, 'US_NHOME_SALES':1,'PPI':1,'CPI':1} #positives shift down, negatives shift up. Goal of this dict is to align data to when it gets released
    
    for entry in offset_dict:
        df[str(entry)] = df[str(entry)].shift(offset_dict[entry])
        
    return df
    
    