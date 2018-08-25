

import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split, GridSearchCV)
import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier

import random
import time
import gc
gc.collect()

#### Get important variables from Boruta #########################
#### For code on how to obtain important variables refer Boruta_Feature_Selection_code.py ###

#### Since running boruta is expensive, we ran the code and obtained 79 important variables for "BUY" model, 88 important variables for "SELL" ,
#### Union of both of them has 96 variables 

selected_buy_3L_121 = pd.read_csv('selected_buy_3L_121.csv')
selected_sell_3L_121 = pd.read_csv('selected_sell_3L_121.csv')
selected_sell_3L_121 = list(selected_sell_3L_121.iloc[:,1])
selected_buy_3L_121 = list(selected_buy_3L_121.iloc[:,1])

#### Read Data for four weeks -- 120,119,118 (Training ) 121 (Testing)
#### The train data is processed and created in R and is being loaded in to Python

xtr121= pd.read_csv('DSG_Train_final_week_121_2.csv')[['IsinIdx','Target_Buy','Target_Sell']+total_selected]

xtr120 = pd.read_csv('DSG_Train_final_week_120_2.csv')[['IsinIdx','Target_Buy','Target_Sell']+total_selected]

xtr119 = pd.read_csv('DSG_Train_final_week_119_2.csv')[['IsinIdx','Target_Buy','Target_Sell']+total_selected]

xtr118 = pd.read_csv('DSG_Train_final_week_118_2.csv')[['IsinIdx','Target_Buy','Target_Sell']+total_selected]

xtr121['week_numeric'] = 120
xtr120['week_numeric'] = 119
xtr119['week_numeric'] = 118
xtr118['week_numeric'] = 117


xtr120 = xtr120.append(xtr119, ignore_index=True)
del xtr119
gc.collect()
xtr120 = xtr120.append(xtr118, ignore_index=True)
del xtr118
gc.collect()


### LIGHTGBM ########
### BUY MODEL ########


test_data_buy=lgb.Dataset(xtr121[selected_buy_3L_121],label=xtr121['Target_Buy'])
train_data_buy=lgb.Dataset(xtr120[selected_buy_3L_121],label=xtr120['Target_Buy'])


num_round= 500
params = {}
params['learning_rate'] = 0.05
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'auc'
#params['sub_feature'] = 0.5
params['num_leaves'] = 30 ### high:overfit but good accuracy 
#params['min_data'] = 50
params['max_depth'] = 6 ### Overfitting
params['max_bin'] = 30  ### high:overfit but good accuracy but slower 
params['lambda_l1'] = 10  
params['lambda_l2'] = 20  
#params['scale_pos_weight'] = 1.1
#params['is_unbalance'] =  True  


lgb_buy_120_121_ub_all = lgb.train(params, train_data_buy,num_round , #nfold=5,
                                       verbose_eval=True,valid_sets=[train_data_buy, test_data_buy],early_stopping_rounds=60, 
                                       valid_names = ['train', 'valid'])

###[500]   train's auc: 0.928539   valid's auc: 0.907095

ax = lgb.plot_importance(lgb_buy_120_121_ub_all, max_num_features=25)


### SELL MODEL ########

test_data_sell=lgb.Dataset(xtr121[selected_sell_3L_121],label=xtr121['Target_Sell'])
train_data_sell=lgb.Dataset(xtr120[selected_sell_3L_121],label=xtr120['Target_Sell'])

num_round= 2000
params = {}
params['learning_rate'] = 0.05
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'auc'
#params['sub_feature'] = 0.5
params['num_leaves'] = 30 ### high:overfit but good accuracy 
#params['min_data'] = 50
params['max_depth'] = 6 ### Overfitting
params['max_bin'] = 30  ### high:overfit but good accuracy but slower 
params['lambda_l1'] = 10  
params['lambda_l2'] = 20  
#params['scale_pos_weight'] = 1.1
#params['is_unbalance'] =  True  

gc.collect()
lgb_sell_120_121_ub_all = lgb.train(params, train_data_sell,500 , #nfold=5,
                                       verbose_eval=True,valid_sets=[train_data_sell, test_data_sell],early_stopping_rounds=60, 
                                       valid_names = ['train', 'valid'])

###[500]   train's auc: 0.910279   valid's auc: 0.885904

ax = lgb.plot_importance(lgb_sell_120_121_ub_all, max_num_features=25)


##########################
### Final Scoring ########
##########################

Test_final = pd.read_csv('Test_final.csv')[['PredictionIdx','BuySell','IsinIdx']+total_selected]
test_sell = Test_final.loc[Test_final["BuySell"]==0]
test_Buy = Test_final.loc[Test_final["BuySell"]==1]


pred_lgb_buy_ub_all =lgb_buy_120_121_ub_all.predict( test_Buy[selected_buy_3L_121])
pred_lgb_sell_ub_all =lgb_sell_120_121_ub_all.predict( test_sell[selected_sell_3L_121])

test_Buy["CustomerInterest"] = pred_lgb_buy_ub_all
test_sell["CustomerInterest"] = pred_lgb_sell_ub_all

final_lgb = test_Buy[['PredictionIdx','CustomerInterest']].append(test_sell[['PredictionIdx','CustomerInterest']], ignore_index=True)

final_lgb.to_csv('repeat_buy_sell_500_lgb.csv',index=False)

### Single Model Scores
###0.78932  Private 
###0.79127  Public 

#############################################
### XGBOOST ########
#############################################

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import gc
gc.collect()


#training XGboost BUY model 
X_train, X_test, y_train, y_test = train_test_split(xtr121[selected_buy_3L_121], xtr121['Target_Buy'], test_size=0.2, random_state=42)
dtrain_buy=xgb.DMatrix(X_train,label= y_train)
dtest_buy=xgb.DMatrix(X_test,label= y_test)

start = time.time() 
parameters={'max_depth':5,'reg_alpha' :5, 'min_child_weight':3,'subsample':0.6,'colsample_bytree':0.8, 'silent':0,'objective':'binary:logistic','eval_metric':'auc','learning_rate':.05, }
model_buy = xgb.train(
    parameters,
    dtrain_buy,
    num_boost_round=500,
    #watchlist = [(dtrain, 'train')],
    evals=[(dtrain_buy, 'train'),(dtest_buy, "Test")],
    early_stopping_rounds=25
)
stop = time.time()
print(stop-start)



#training XGboost SELL model 
X_train, X_test, y_train, y_test = train_test_split(xtr121[selected_sell_3L_121], xtr121['Target_Sell'], test_size=0.2, random_state=42)
dtrain=xgb.DMatrix(X_train,label= y_train)
dtest=xgb.DMatrix(X_test,label= y_test)

start = time.time() 
parameters={'max_depth':5,'reg_alpha' :5, 'min_child_weight':3,'subsample':0.6,'colsample_bytree':0.8, 'silent':0,'objective':'binary:logistic','eval_metric':'auc','learning_rate':.05 }
model_sell = xgb.train(
    parameters,
    dtrain,
    num_boost_round=500,
    #watchlist = [(dtrain, 'train')],
    evals=[(dtrain, 'train'),(dtest, "Test")],
    early_stopping_rounds=50
)
stop = time.time()
print(stop-start)




#### SCORING ### 

pred_xgb_buy =model_buy.predict(xgb.DMatrix(test_Buy[selected_buy_3L_121]))
pred_xgb_sell =model_sell.predict(xgb.DMatrix(test_sell[selected_sell_3L_121]))
pd.Series(pred_xgb_buy).describe()
test_Buy["CustomerInterest"] = pred_xgb_buy
test_sell["CustomerInterest"] = pred_xgb_sell

final_xgb = test_Buy[['PredictionIdx','CustomerInterest']].append(test_sell[['PredictionIdx','CustomerInterest']], ignore_index=True)

final_xgb.to_csv('buy_sell_seperate_xgb_500.csv',index=False)


#### single  XGBOOST model LB scores 
### 0.77473 Private
### 0.77883 Public 

##################################################################################################
####################################################################################################
###################################################################################################
#########################################################
#### LIGHT GBM with additional market variables #########
#########################################################
del X_train,X_test,y_train,y_test
gc.collect()
### Load market related additional variables created using R #######

gc.collect()

Market_related_new = pd.read_csv('Market_related_new.csv')
Market_related_test = pd.read_csv('Market_related_test.csv')

xtr121 = pd.merge(xtr121,Market_related_new[['IsinIdx','week_numeric','Price',
 'Yield',
 #'sd_Yield',
 #'sd_z',
 'Coupon_amount',
 'ZSpread',
 'Timetomature',
 #'semi_annual_periods',
 'near_coupon_date',
 'perc_yield',
 'perc_price',
 'perc_coupon',
 'duration']],how='left', on=['IsinIdx','week_numeric'])

xtr120 = pd.merge(xtr120,Market_related_new[['IsinIdx','week_numeric','Price',
 'Yield',
 #'sd_Yield',
 #'sd_z',
 'Coupon_amount',
 'ZSpread',
 'Timetomature',
 #'semi_annual_periods',
 'near_coupon_date',
 'perc_yield',
 'perc_price',
 'perc_coupon',
 'duration']],how='left', on=['IsinIdx','week_numeric'])


xtr121 = xtr121.fillna(-999)
xtr120 = xtr120.fillna(-999)
#del Market_related_new

gc.collect()
new_selected = ['Price',
 'Yield',
 #'sd_Yield',
 #'sd_z',
 'Coupon_amount',
 'ZSpread',
 'Timetomature',
 #'semi_annual_periods',
 'near_coupon_date',
 'perc_yield',
 'perc_price',
 'perc_coupon',
 'duration']

############ BUY ############## LIGHT GBM ##################
#del Test_final_market,test_sell_market,test_buy_market
#del finalss3
#del final_lgb_market,final_lgb_market2
#del test_Buy,test_sell
#del finalss1
#del pred_lgb_buy_ub_all,pred_lgb_sell_ub_all,pred_xgb_buy,pred_xgb_sell
gc.collect()


test_data_buy=lgb.Dataset(xtr121[selected_buy_3L_121 + new_selected],label=xtr121['Target_Buy'])
train_data_buy=lgb.Dataset(xtr120[selected_buy_3L_121+ new_selected],label=xtr120['Target_Buy'])

num_round= 500
params = {}
params['learning_rate'] = 0.05
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'auc'
#params['sub_feature'] = 0.5
params['num_leaves'] = 30 ### high:overfit but good accuracy 
#params['min_data'] = 50
params['max_depth'] = 6 ### Overfitting
params['max_bin'] = 30  ### high:overfit but good accuracy but slower 
params['lambda_l1'] = 10  
params['lambda_l2'] = 20  
#params['scale_pos_weight'] = 1000
params['is_unbalance'] =  True  

gc.collect()
lgb_buy_market = lgb.train(params, train_data_buy,500 , #nfold=5,
                               verbose_eval=True,valid_sets=[train_data_buy, test_data_buy],early_stopping_rounds=60, 
                               valid_names = ['train', 'valid'])

ax = lgb.plot_importance(lgb_buy_market, max_num_features=30)
############ SELL ############## LIGHT GBM ##################
del test_data_buy,train_data_buy
gc.collect()

test_data_sell=lgb.Dataset(xtr121[selected_sell_3L_121+new_selected],label=xtr121['Target_Sell'])
train_data_sell=lgb.Dataset(xtr120[selected_sell_3L_121+new_selected],label=xtr120['Target_Sell'])

gc.collect()

lgb_sell_market = lgb.train(params, train_data_sell,500 , #nfold=5,
                                 verbose_eval=True,valid_sets=[train_data_sell, test_data_sell],early_stopping_rounds=30, 
                                 valid_names = ['train', 'valid'])
ax = lgb.plot_importance(lgb_sell_market, max_num_features=30)

del test_data_sell,train_data_sell
gc.collect()
########### Scoring ############### 
Test_final_market = pd.merge(Test_final,Market_related_test[['IsinIdx','Price',
                                                      'Yield',
                                                      #'sd_Yield',
                                                      #'sd_z',
                                                      'Coupon_amount',
                                                      'ZSpread',
                                                      'Timetomature',
                                                      #'semi_annual_periods',
                                                      'near_coupon_date',
                                                      'perc_yield',
                                                      'perc_price',
                                                      'perc_coupon',
                                                      'duration']],how='left', on=['IsinIdx'])


Test_final_market = Test_final_market.fillna(-999)

test_sell_market = Test_final_market.loc[Test_final_market["BuySell"]==0]

test_buy_market = Test_final_market.loc[Test_final_market["BuySell"]==1]

pred_lgb_buy_market =lgb_buy_market.predict( test_buy_market[selected_buy_3L_121+new_selected])
pred_lgb_sell_market =lgb_sell_market.predict( test_sell_market[selected_sell_3L_121+new_selected])
test_buy_market["CustomerInterest"] = pred_lgb_buy_market
test_sell_market["CustomerInterest"] = pred_lgb_sell_market
final_lgb_market_500 = test_buy_market[['PredictionIdx','CustomerInterest']].append(test_sell_market[['PredictionIdx','CustomerInterest']], ignore_index=True)

final_lgb_market_500.to_csv('buy_sell_seperate_lgb_500_market_ub.csv',index=False)
########### Same model 200 rounds 


test_data_buy=lgb.Dataset(xtr121[selected_buy_3L_121 + new_selected],label=xtr121['Target_Buy'])
train_data_buy=lgb.Dataset(xtr120[selected_buy_3L_121+ new_selected],label=xtr120['Target_Buy'])

num_round= 200
params = {}
params['learning_rate'] = 0.05
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'auc'
#params['sub_feature'] = 0.5
params['num_leaves'] = 30 ### high:overfit but good accuracy 
#params['min_data'] = 50
params['max_depth'] = 6 ### Overfitting
params['max_bin'] = 30  ### high:overfit but good accuracy but slower 
params['lambda_l1'] = 10  
params['lambda_l2'] = 20  
#params['scale_pos_weight'] = 1000
params['is_unbalance'] =  True  

gc.collect()
lgb_buy_market = lgb.train(params, train_data_buy,200 , #nfold=5,
                               verbose_eval=True,valid_sets=[train_data_buy, test_data_buy],early_stopping_rounds=60, 
                               valid_names = ['train', 'valid'])

ax = lgb.plot_importance(lgb_buy_market, max_num_features=30)
############ SELL ############## LIGHT GBM ##################
del test_data_buy,train_data_buy
gc.collect()

test_data_sell=lgb.Dataset(xtr121[selected_sell_3L_121+new_selected],label=xtr121['Target_Sell'])
train_data_sell=lgb.Dataset(xtr120[selected_sell_3L_121+new_selected],label=xtr120['Target_Sell'])

gc.collect()

lgb_sell_market = lgb.train(params, train_data_sell,200 , #nfold=5,
                                 verbose_eval=True,valid_sets=[train_data_sell, test_data_sell],early_stopping_rounds=30, 
                                 valid_names = ['train', 'valid'])
ax = lgb.plot_importance(lgb_sell_market, max_num_features=30)

del test_data_sell,train_data_sell
gc.collect()
########### Scoring ############### 
Test_final_market = pd.merge(Test_final,Market_related_test[['IsinIdx','Price',
                                                      'Yield',
                                                      #'sd_Yield',
                                                      #'sd_z',
                                                      'Coupon_amount',
                                                      'ZSpread',
                                                      'Timetomature',
                                                      #'semi_annual_periods',
                                                      'near_coupon_date',
                                                      'perc_yield',
                                                      'perc_price',
                                                      'perc_coupon',
                                                      'duration']],how='left', on=['IsinIdx'])


Test_final_market = Test_final_market.fillna(-999)

test_sell_market = Test_final_market.loc[Test_final_market["BuySell"]==0]

test_buy_market = Test_final_market.loc[Test_final_market["BuySell"]==1]

pred_lgb_buy_market =lgb_buy_market.predict( test_buy_market[selected_buy_3L_121+new_selected])
pred_lgb_sell_market =lgb_sell_market.predict( test_sell_market[selected_sell_3L_121+new_selected])
test_buy_market["CustomerInterest"] = pred_lgb_buy_market
test_sell_market["CustomerInterest"] = pred_lgb_sell_market
final_lgb_market_200 = test_buy_market[['PredictionIdx','CustomerInterest']].append(test_sell_market[['PredictionIdx','CustomerInterest']], ignore_index=True)

final_lgb_market_200.to_csv('buy_sell_seperate_lgb_500_market_ub.csv',index=False)

############## SINGLE LGB MODEL with market variables LB Scores ############
#### 
####0.78922 ##Private , 0.79017 ## Public @@@200 rounds
####0.78868 ##Private , 0.78950 ## Public @@@500 rounds
############################################################################

#### Final Ensembling of models #####


#final_lgb
#final_xgb
#final_lgb_market_200
#final_lgb_market_500


import scipy.stats as ss

def min_max_scaler (X):
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X_std

## buy_sell_including_market_try1.csv
final_lgb = pd.read_csv('repeat_buy_sell_500_lgb.csv')
final_xgb = pd.read_csv('buy_sell_seperate_xgb_500.csv')
final_lgb_market_200 = pd.read_csv('buy_sell_seperate_lgb_200_market_ub.csv')
final_lgb_market_500 = pd.read_csv('buy_sell_seperate_lgb_500_market_ub.csv')

final_lgb = final_lgb.sort_values(by = ['PredictionIdx'])
final_xgb = final_xgb.sort_values(by = ['PredictionIdx'])
final_lgb_market_200 = final_lgb_market_200.sort_values(by = ['PredictionIdx'])
final_lgb_market_500 = final_lgb_market_500.sort_values(by = ['PredictionIdx'])

### Sanity checks
np.sum(final_lgb.PredictionIdx != final_xgb.PredictionIdx ) ## 0
np.sum(final_lgb_market_200.PredictionIdx != final_lgb_market_500.PredictionIdx ) ## 0
np.sum(final_lgb.PredictionIdx != final_lgb_market_500.PredictionIdx ) ## 0

### Weighted ranked ensembling 

final_sub = final_lgb
final_sub['CustomerInterest'] = min_max_scaler(0.75*(0.33*ss.rankdata(final_lgb.CustomerInterest) + 0.33*ss.rankdata(final_lgb_market_200.CustomerInterest) + 0.33*ss.rankdata(final_lgb_market_500.CustomerInterest))+ 0.25*ss.rankdata(final_xgb.CustomerInterest)) 
final_sub.to_csv('submission_ensemble.csv',index = False)

### 0.79452 Private
### 0.79640 Public

### The LB score is further improved by ensembling the above models with different seeds and different parameters 
### But the complexity and run time increases heavily for a marginal gain 
#########################################################################################










