import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

#returns a table of probabilities for the testing data

def create_probs(X,y,start,end,split,time):
    """
    Returns a table of probabilities for the testing data and time horizons being tested. Arguments are as follows:
    
    X: full X data
    y: full y data
    start: start of the dataset
    end: end of the dataset
    split: index position of where you want to split your train and test
    time: list of horizons you're testing
   
    """
    
    result = pd.DataFrame(y.iloc[split:end].values)
    result.index = y.iloc[split:end].index
    
    eval_metrics = pd.DataFrame(columns = ['Time_Horizon','AUC','Log_Loss'])

    for point in time:
    
        model = RandomForestClassifier(n_estimators=500, max_depth=4, random_state=45, max_features = 'sqrt')
    
    
        y_shift = y.shift(point) #needs to be negative to look forward
        y_shift = y_shift.fillna(0)
    
        X_train = X.iloc[start:split]
        X_test = X.iloc[split:end]
        y_train = y_shift.iloc[start:split]
        y_test = y_shift.iloc[split:end]
    

        model.fit(X_train, y_train)
    
        probs = model.predict_proba(X_test)
        
        evals = {'Time_Horizon': point,
                 'AUC': roc_auc_score(y_test.values, probs[:,1:]),
                 'Log_Loss': log_loss(y_test, probs)}
        
        eval_metrics = eval_metrics.append(evals, ignore_index = True)
    
        result[str(point)] = probs[:,1]
        
    result = result.drop(columns = 0)
    result.columns = ['Current Month','1 Month Horizon','3 Month Horizon','12 Month Horizon']
    
    return result, eval_metrics

def create_chart(probs_chart, column = 0):
    
    """
    Takes in probability chart and the specific series you want to graph then graphs it with 2005+ recessions plotted
    """
    
    x = probs_chart.index
    series = probs_chart.iloc[:,column]

    fig, ax = plt.subplots(figsize=(18, 6))

    ax.plot(x, series, linewidth=2.5, zorder=2)
    ax.scatter(x, series, s=0, zorder=1, label='_nolegend_')
    ax.axvspan(x[38],x[56], color=sns.xkcd_rgb['grey'], alpha=0.5)
    ax.axvspan(x[74],x[77], color=sns.xkcd_rgb['grey'], alpha=0.5)
    ax.axvspan(x[80],x[83], color=sns.xkcd_rgb['grey'], alpha=0.5)

    ax.set_title('Recession Prediction With Contractions in Real GDP Shaded in Gray', fontsize=14, fontweight='demi')

    ax.legend(loc='upper left', fontsize=11, frameon=True).get_frame().set_edgecolor('blue')

    ax.set_ylabel('% Probability of Q/Q Decrease in Real GDP')
    ax.set_xlabel('Date')