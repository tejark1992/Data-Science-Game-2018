TEAM - DATA EMISSARIES 
COLLEGE - IIM CALCUTTA
COUNTRY - INDIA
PRIVATE LB RANK - 10


File 1: 
We have used R for data preprocessing and Train & Test data creation including all the features to be used in subsequent models
Running the entire code will create the following files which are used later.
i) Market_related_new.csv
ii) Market_related_test.csv
iii) DSG_Train_final_week_121_2.csv
iv) DSG_Train_final_week_120_2.csv
v) DSG_Train_final_week_119_2.csv
vi) DSG_Train_final_week_118_2.csv
vii) Test_final.csv

File 2: 
We have used BorutaPy for selecting important features from those created in step-1. 
we ran the code and obtained 79 important variables for "BUY" model, 88 important variables for "SELL" model.
The following two files (lists of important feature names) are created at the end of the code.
Since running boruta is expensive, We have attached them along with code files so that running this code is made Optional. 
i) selected_buy_3L_121.csv
ii)selected_sell_3L_121.csv


File 3:
It has code for 3 LightGBM + 1 Xgboost models each for BUY and Sell (Total of 8 models), which are ensembled at the end of the code.

#####
eof






 










