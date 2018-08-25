
#### OPTIONAL TO RUN ######
#### Important variable selection using BorutaPy
 
import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split, GridSearchCV)
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import random
import time
from boruta import BorutaPy
rfc = RandomForestClassifier(n_estimators=200, n_jobs=4, class_weight='balanced', max_depth=6)

#### Load Data #####
xtr121 = pd.read_csv('DSG_Train_final_week_121_2.csv')

#xtr121.iloc[:,240].describe()
rm = ['CustomerIdx','IsinIdx','cust.bond','Target_Buy','Target_Sell','Sell_3',
 'Buy_2',
 'Buy_5',
 'Sell_4',
 'Buy_1',
 'Buy_3',
 'Sell_2',
 'Sell_1',
 'Sell_5',
 'Buy_4']

n = 300000
s = random.sample(range(0,xtr121.shape[0]), n)

X = xtr121.iloc[s,:].drop(rm, axis=1).values
y = xtr121.iloc[s,:]['Target_Buy'].values

boruta_selector_buy = BorutaPy(rfc, n_estimators="auto", verbose=2,max_iter = 25)
start_time = time.time()
boruta_selector_buy.fit(X, y)
print(time.time() - start_time)

y = xtr121.iloc[s,:]['Target_Sell'].values
boruta_selector_sell = BorutaPy(rfc, n_estimators="auto", verbose=2,max_iter = 25)
start_time = time.time()
boruta_selector_sell.fit(X, y)
print(time.time() - start_time)

### Get the selected list of variables ############
selected_buy_3L_121 = xtr121.iloc[s,:].drop(rm, axis=1).columns[boruta_selector_buy.support_]
selected_sell_3L_121 = xtr121.iloc[s,:].drop(rm, axis=1).columns[boruta_selector_sell.support_]
#pd.Series(boruta_selector_buy.ranking_).to_csv('ranking_buy_3L_121.csv')
#pd.Series(boruta_selector_sell.ranking_).to_csv('ranking_sell_3L_121.csv')

#### and save them ########
pd.Series(selected_buy_3L_121).to_csv('selected_buy_3L_121.csv')
pd.Series(selected_sell_3L_121).to_csv('selected_sell_3L_121.csv')










