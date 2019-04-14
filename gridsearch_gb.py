import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import xgboost as xgb


max_depth = [1,2,3,4,5,7,10]
n_est = [100, 200, 300,500, 1000,1500,2000]
learning_rate = [0.001,0.005,0.010]

#takes in 3 hyperparameters in list form as well as split X and Y data and outputs a pandas dataframe with those list parameters and the corresponding log loss/AUC

def grid_search(max_depth, n_est, learning_rate, X_train, y_train, X_test, y_test):

    hyperparams = pd.DataFrame(columns = ['Max_Depth','N_Est','Learning_Rate','Log_Loss','AUC'])


    for depth in max_depth:
        for n in n_est:
            for r in learning_rate:

                model = xgb.XGBClassifier(learning_rate=r,
                                       n_estimators=n, #bump this and learning rate to make more trees, trees(not exact) = n*learning rate
                                       min_samples_leaf=4,
                                       max_depth=depth,
                                       subsample=0.5)
                model.fit(X_train, y_train) #fitting model
                probs = model.predict_proba(X_test)
                ll = log_loss(y_test, probs)
                auc = roc_auc_score(y_test.values, probs[:,1:])
            
                hyps = {'Max_Depth': depth,
                                'N_Est':n,
                                'Learning_Rate':r,
                                'Log_Loss':ll,
                                'AUC':auc}
            
                hyperparams = hyperparams.append(hyps, ignore_index=True)
                
    return hyperparams