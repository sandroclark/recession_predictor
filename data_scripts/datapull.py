#imports

import pandas as pd

import quandl
quandl.ApiConfig.api_key = 'm8FYMyoCaJSbTrBASNHh'

#pull data from quandl

def quandl_pull(source_csv = './data/datasources2.csv'):
    
    data = pd.read_csv(source_csv)
    cols = list(data['Var_name'].astype('str'))
    dataset = quandl.get([val for val in data['Quandl Key']]) #looping through the QUANDL keys to pull it into one DF
    dataset.columns = cols
    
    
    #special logic to factor in for 3YRT being a daily
    treas = dataset['3YRT']
    dataset = dataset.drop(columns = ['3YRT'])
    treas = treas.resample('MS').mean()
    
    
    #now that daily 3YR treas is out of the dataset, converting data set to monthly
    dataset.index = dataset.index.strftime('%Y-%m') #converting the datetime index to Y/M so it is collapsable
    dataset = dataset.groupby(dataset.index, as_index=True).agg(sum) #collapsing by Y/M
    
    #readd 3YRT back into data
    dataset = dataset.join(treas, how='outer')
    
    return dataset

#pull yield curve info. this comes from the Fed website as they have historicals that QUANDL doesn't have
#https://www.newyorkfed.org/research/capital_markets/ycfaq.html
def yieldcurve_pull(source_csv = './data/Fed10Y_3M.csv'):
    
    yields = pd.read_csv(source_csv)
    yields['Date'] = pd.to_datetime(yields['Date'])
    mask = (yields['Date'] > pd.datetime(2058,1,1))
    yields.loc[mask,'Date'] = yields.loc[mask,'Date'].apply(lambda x: x - pd.DateOffset(years=100))

    yields['Date'] = yields['Date'].apply(lambda x: x.strftime('%Y-%m'))

    yields = yields.set_index('Date')
    yields = yields.drop(['3 Month Treasury Yield', 'Rec_prob', 'NBER_Rec'], axis=1)
    
    return yields