
setwd("C:/Users/Yash/Desktop/dsg")
gc()
rm(list = ls())

library(data.table)
library(plyr)
library(xgboost)
library(readr)
library(dplyr)
library(readr)
library(ggplot2)
library(sqldf)


### Reading Transaction file ###
Trade <- fread("Trade.csv")
Challenge_20180423 <- fread("Challenge_20180423.csv")
Trade$TradeDateKey = as.Date(as.character(Trade$TradeDateKey), format = '%Y%m%d')
Trade$week = as.numeric(paste(format(Trade$TradeDateKey,'%y'),format(Trade$TradeDateKey,"%V"),sep = ''))
Trade$week = replace(Trade$week,Trade$week==1653,1553)

######### get only customers & bonds in the test data #######
cs = unique(Challenge_20180423$CustomerIdx)
cs = cs[order(cs)]

#length(cs) ### 2495
is = unique(Challenge_20180423$IsinIdx)
is = is[order(is)]
#length(is) ### 18265

## Data cleaning Trade table

Trade = Trade[Trade$CustomerIdx %in% cs | Trade$IsinIdx %in% is, ]
Trade$week_numeric = as.numeric(as.factor(Trade$week))
Trade$cust.bond = as.character(paste(Trade$CustomerIdx,Trade$IsinIdx,sep ='|'))
Trade$Price2 = replace(Trade$Price, (Trade$Price <0 |Trade$Price >200 | is.na(Trade$Price)),-999 )
Trade$Transaction_Year <- format(Trade$TradeDateKey, "%Y")
Trade$Transaction_Month <- format(Trade$TradeDateKey, "%Y-%m")
Trade$quarter <- quarters(Trade$TradeDateKey)

### selecting only approximately last 6-7  month transaction data 

###Trade <- Trade[Trade$week_numeric>90,]


## Add other customer & bond related variables #####
## customer related ####


Customer <- read_csv("Customer.csv")
Isin <- read_csv("Isin.csv")
Isin$Seniority_new = replace(Isin$Seniority,Isin$Seniority %in% c('ASS','COL'),'MOR')
Isin$Currency_new = replace(Isin$Currency,!Isin$Currency %in% c('USD','EUR','GBP','CHF','CNO','CNH'),'OTHERS')
Isin$IndustrySector_new = replace(Isin$IndustrySector,Isin$IndustrySector %in% c('Asset Backed Securit','Mortgage Securities'),'Diversified')

########################

selected_columns <- c('IsinIdx','ActivityGroup','Region','CompositeRating','IndustrySector_new','CouponType','MarketIssue',
                      'Seniority_new','Currency_new','ActualMaturityDateKey','IssueDateKey')

Isin$ActualMaturityDateKey = as.Date(as.character(Isin$ActualMaturityDateKey), format = '%Y%m%d')

Isin$IssueDateKey = as.Date(as.character(Isin$IssueDateKey), format = '%Y%m%d')

Trade = left_join(Trade,Isin[,selected_columns],'IsinIdx')

Trade = left_join(Trade,Customer[,c(1,2,4)],'CustomerIdx')


####################################
##### It takes only holding transactions for weeks < w and 
##### create aggregate features for each customer&bond pair



####################################

Variable_creation_func_holding = function(Tr,w){
  X_build_1 <- Tr[Tr$CustomerInterest==0 & Tr$week_numeric < w,]
  
  X_build_1 = data.table(X_build_1)
  X_month_1 <- X_build_1[, .(Count = .N), .(CustomerIdx, IsinIdx, Transaction_Month)]
  X_week_1 <- X_build_1[, .(Count = .N), .(CustomerIdx, IsinIdx, week)]
  
  X_count_1 <- X_month_1[, .(Count = .N), .(CustomerIdx, IsinIdx)]
  X_count_week_1 <- X_week_1[, .(Count = .N), .(CustomerIdx, IsinIdx)]
  
  
  X_buysell_1 <- dcast(X_build_1, CustomerIdx+ IsinIdx ~ BuySell, length, value.var="TradeDateKey", fill=0)
  
  #X_bs_ts_1 <- dcast(X_build_1, CustomerIdx + IsinIdx ~ BuySell, mean, value.var="NotionalEUR", fill=0)
  
  customer_bond_hold <- X_build_1[,.(Distinct_holdings_Cb = length(unique(NotionalEUR)),sd_hold = sd(NotionalEUR),mean_hold= mean(NotionalEUR)),by = c('CustomerIdx','IsinIdx')]
  customer_bond_hold[is.na(customer_bond_hold)] <- 0 
  
  ##### Date since last 
  max_d = max(X_build_1$TradeDateKey)
  X_date_1 <- X_build_1[, .(Max_Date = max(TradeDateKey)), .(CustomerIdx , IsinIdx)]
  X_date_1 <- X_date_1[, ":="(Last_Date_holded = as.numeric(max_d - Max_Date),
                              Max_Date = NULL)]
  
  X_date_2 <- X_build_1[, .(Max_Date = max(ActualMaturityDateKey)), .(CustomerIdx , IsinIdx)]
  X_date_2 <- X_date_2[, ":="(Time_to_maturity_hold = as.numeric(Max_Date - max_d),
                              Max_Date = NULL)]
  X_train_1 <- Reduce(function(x, y) merge(x, y, all=T, by=c("CustomerIdx" , "IsinIdx")),
                      list(X_count_1, X_count_week_1,
                           X_buysell_1,customer_bond_hold, X_date_1,X_date_2))
  
  X_train_1
}


Variable_creation_func_both = function(Tr,w){
  X_build_1 <- Tr[Tr$week_numeric < w,]
  
  X_build_1 = data.table(X_build_1)
  
  X_ActivityGroup <- dcast(X_build_1, CustomerIdx + IsinIdx ~ BuySell+ActivityGroup, length, value.var="TradeDateKey", fill=0)
  
  X_CompositeRating <- dcast(X_build_1, CustomerIdx + IsinIdx ~ BuySell+CompositeRating, length, value.var="TradeDateKey", fill=0)
  
  X_Region.x <- dcast(X_build_1, CustomerIdx + IsinIdx ~ BuySell+Region.x, length, value.var="TradeDateKey", fill=0)
  
  X_IndustrySector_new <- dcast(X_build_1, CustomerIdx + IsinIdx ~ BuySell+IndustrySector_new, length, value.var="TradeDateKey", fill=0)
  
  X_CouponType <- dcast(X_build_1, CustomerIdx + IsinIdx ~ BuySell+CouponType, length, value.var="TradeDateKey", fill=0)
  
  X_MarketIssue <- dcast(X_build_1, CustomerIdx + IsinIdx ~ BuySell+MarketIssue, length, value.var="TradeDateKey", fill=0)
  
  X_Seniority_new <- dcast(X_build_1, CustomerIdx + IsinIdx ~ BuySell+ Seniority_new, length, value.var="TradeDateKey", fill=0)
  
  X_Currency_new <- dcast(X_build_1, CustomerIdx + IsinIdx ~ BuySell+ Currency_new, length, value.var="TradeDateKey", fill=0)
  
  X_Sector <- dcast(X_build_1, CustomerIdx + IsinIdx ~ BuySell+ Sector, length, value.var="TradeDateKey", fill=0)
  
  X_Region.y <- dcast(X_build_1, CustomerIdx + IsinIdx ~ BuySell+ Region.y, length, value.var="TradeDateKey", fill=0)
  
  X_train_1 <- Reduce(function(x, y) merge(x, y, all=T, by=c("CustomerIdx" , "IsinIdx")),
                      list(#X_count_1, X_count_week_1, X_buysell_1,
                        #X_ts_1, X_bs_ts_1, X_bs_ts_2,X_bs_ts_2b, X_bs_ts_3, X_date_1,
                        #X_quarter,
                        X_ActivityGroup,
                        X_CompositeRating,
                        X_Region.x,
                        X_IndustrySector_new,
                        X_CouponType,
                        X_MarketIssue,
                        X_Seniority_new,
                        X_Currency_new,
                        X_Sector,
                        X_Region.y#,
                        #X_date_2
                      ))
  
}







### All non holding transactions ### creation of aggregate variables for cust.bond pair ###

Variable_creation_func1 = function(Tr,w){
  X_build_1 <- Tr[Tr$CustomerInterest==1 & Tr$week_numeric < w,]
  X_build_1 <- data.table(X_build_1)
  X_build_1$week_numeric_2 = w - X_build_1$week_numeric
  X_build_1$week_numeric_2 = replace(X_build_1$week_numeric_2,X_build_1$week_numeric_2>5,5)
  
  X_month_1 <- X_build_1[, .(Count = .N), .(CustomerIdx , IsinIdx, Transaction_Month)]
  X_week_1 <- X_build_1[, .(Count = .N), .(CustomerIdx , IsinIdx, week_numeric)]
  
  X_count_month_1 <- X_month_1[, .(Count = .N), .(CustomerIdx , IsinIdx)]
  X_count_week_1 <- X_week_1[, .(Count = .N,count_1 = sum(week_numeric>=w-1),count_6 = sum(week_numeric>=w-6),count_12 = sum(week_numeric>=w-12),count_24 = sum(week_numeric>=w-24)), .(CustomerIdx , IsinIdx)]
  
  ### Buysell counts 
  X_buysell_1 <- dcast(X_build_1, CustomerIdx + IsinIdx ~ BuySell, length, value.var="TradeDateKey", fill=0)
  #X_ts_1 <- dcast(X_build_1, CustomerIdx + IsinIdx ~ TradeStatus, length, value.var="TradeDateKey", fill=0)
  
  ### Buysell + trade status counts
  X_bs_ts_1 <- dcast(X_build_1, CustomerIdx + IsinIdx ~ BuySell+TradeStatus, length, value.var="TradeDateKey", fill=0)
  
  ### Buysell + trade status mean & SD notional EUR 
  X_bs_ts_2 <- dcast(X_build_1, CustomerIdx + IsinIdx ~ BuySell+TradeStatus, mean, value.var="NotionalEUR", fill=0)
  
  #X_bs_ts_2b <- dcast(X_build_1, CustomerIdx + IsinIdx ~ BuySell+TradeStatus, sd, value.var="NotionalEUR", fill=0)
  
  X_bs_ts_3 <- dcast(X_build_1, CustomerIdx + IsinIdx ~ BuySell+TradeStatus, median, value.var="Price2", fill=0)
  
  X_quarter <- dcast(X_build_1, CustomerIdx + IsinIdx ~ BuySell+quarter, length, value.var="TradeDateKey", fill=0)
  X_week_numeric <- dcast(X_build_1, CustomerIdx + IsinIdx ~ BuySell+week_numeric_2, length, value.var="TradeDateKey", fill=0)
  
  
  ### Last Transaction whether buy or sell??
  library(dplyr)
  
  X_lag_bs <- unique(X_build_1[,c('CustomerIdx','IsinIdx','TradeDateKey','BuySell')])
  X_lag_bs = X_lag_bs[,.(c_bs =.N,c_b = sum(BuySell=='Buy') ),c('CustomerIdx','IsinIdx','TradeDateKey')]
  X_lag_bs$BuySell = ifelse(X_lag_bs$c_bs>1,2,X_lag_bs$c_b)
  
  X_lag_bs<-X_lag_bs[,list(BuySell,TradeDateKey,max_trade_date=max(TradeDateKey)),by=c("CustomerIdx","IsinIdx")]
  
  X_lag_bs1= X_lag_bs[, lag.BuySell:=c(NA, BuySell[-.N]), by=c(c('CustomerIdx','IsinIdx','max_trade_date'))] %>% filter(max_trade_date == TradeDateKey)
  
  X_lag_bs1$lag.BuySell = ifelse(is.na(X_lag_bs1$lag.BuySell),-99,X_lag_bs1$lag.BuySell ) 
  X_lag_bs1 = data.table(X_lag_bs1[,c(1,2,6)])
  max_d = max(X_build_1$TradeDateKey)
  X_date_0_buy <- X_build_1[BuySell == 'Buy', .(Max_Date = max(TradeDateKey)), .(CustomerIdx , IsinIdx)]
  X_date_0_buy <- X_date_0_buy[, ":="(Last_Date_buy = as.numeric(max_d - Max_Date),
                                      Max_Date = NULL)]
  
  X_date_0_sell <- X_build_1[BuySell == 'Sell', .(Max_Date = max(TradeDateKey)), .(CustomerIdx , IsinIdx)]
  X_date_0_sell <- X_date_0_sell[, ":="(Last_Date_sell = as.numeric(max_d - Max_Date),
                                        Max_Date = NULL)]
  
  X_date_1 <- X_build_1[, .(Max_Date = max(TradeDateKey)), .(CustomerIdx , IsinIdx)]
  
  X_date_1 <- X_date_1[, ":="(Last_Date = as.numeric(max_d - Max_Date),
                              Max_Date = NULL)]
  X_date_2 <- X_build_1[, .(Max_Date = max(ActualMaturityDateKey)), .(CustomerIdx , IsinIdx)]
  X_date_2 <- X_date_2[, ":="(Last_Date_maturity = as.numeric(Max_Date - max_d),
                              Max_Date = NULL)]
  X_date_3 <- X_build_1[, .(Max_Date = max(IssueDateKey)), .(CustomerIdx , IsinIdx)]
  X_date_3 <- X_date_3[, ":="(Last_Date_issue = as.numeric(max_d - Max_Date ),
                              Max_Date = NULL)]
  X_train_1 <- Reduce(function(x, y) merge(x, y, all=T, by=c("CustomerIdx" , "IsinIdx")),
                      list(X_count_month_1, X_count_week_1, X_buysell_1,
                           X_bs_ts_1, X_bs_ts_2, X_bs_ts_3,
                           X_lag_bs1,
                           X_date_1,
                           X_date_2
                           ,X_date_3
                           ,X_date_0_sell
                           ,X_date_0_buy
                           ,X_quarter
                           ,X_week_numeric
                      ))
  
  X_train_1$Issue_mature_ratio = X_train_1$Last_Date_issue/(X_train_1$Last_Date_issue + X_train_1$Last_Date_maturity)
  X_train_1$Last_transac_ratio = X_train_1$Last_Date/(X_train_1$Last_Date_issue + X_train_1$Last_Date_maturity)
  #X_train_1$Last_transac_ratio_2 = X_train_1$Last_Date/(X_train_1$Last_Date_issue)
  
  X_train_1[is.na(X_train_1)] <- -999
  
  X_train_1
}





############## Only Customer related variables ################


Variable_creation_customer = function(Trade_temp,customer_week)
{
  Trade_temp2 = Trade_temp[Trade_temp$CustomerInterest == 1,]
  
  Trade_temp2 = data.table(Trade_temp2)
  
  customer_1 = data.table(Trade_temp)[,.(N_distinct_bonds_c = length(unique(IsinIdx))),by = c('CustomerIdx')]
  customer_2 = data.table(Trade_temp2)[,.(N_distinct_bond_c_bs = length(unique(IsinIdx)),avg_nom = mean(NotionalEUR)),by = c('CustomerIdx','BuySell')]
  c_bs <- as.data.frame(dcast(Trade_temp2, CustomerIdx ~ BuySell, value.var = 'CustomerInterest'))
  c_ts <- as.data.frame(dcast(Trade_temp2, CustomerIdx ~TradeStatus , value.var = 'CustomerInterest'))
  c_ts$total = rowSums(c_ts[,-1])
  for (i in c(2,4,5))
  {
    c_ts[,i] =   c_ts[,i]/ c_ts$total
  }
  c_ts = c_ts[,-c(3)]
  print('Hoorah"')
  
  Customer_holding_info = data.table(Trade_temp[Trade_temp$CustomerInterest ==0,])[,.(total_holding_counts = .N,distinct_holding_counts = length(unique(NotionalEUR)), distinct_bonds_holded = length(unique(IsinIdx)), distcint_weeks_declared = length(unique(week))),by = c('CustomerIdx')]
  all_customer = left_join(customer_1,customer_2[customer_2$BuySell=='Buy',],by = 'CustomerIdx')
  all_customer = left_join(all_customer,customer_2[customer_2$BuySell=='Sell',],by = 'CustomerIdx')
  all_customer[is.na(all_customer)] = 0
  all_customer = left_join(all_customer,c_bs,by = 'CustomerIdx')
  all_customer = all_customer[,-c(3,6)]
  
  all_customer = left_join(all_customer,c_ts,by = 'CustomerIdx')
  all_customer = left_join(all_customer,Customer_holding_info,by = 'CustomerIdx')
  
  all_customer = left_join(all_customer,customer_week,by = 'CustomerIdx')
  
  all_customer[is.na(all_customer)] = -999
  
  
  
  #X_lag_bs <- Trade_temp2[,c('CustomerIdx','TradeDateKey','BuySell')]
  #X_lag_bs<-X_lag_bs[,list(BuySell,TradeDateKey,max_trade_date=max(TradeDateKey)),by=c("CustomerIdx")]
  
  #X_lag_bs1= X_lag_bs[, lag.BuySell_cust:=c(NA, BuySell[-.N]), by=c(c('CustomerIdx','max_trade_date'))] %>% filter(max_trade_date == TradeDateKey)
  
  #X_lag_bs1$lag.BuySell = ifelse(is.na(X_lag_bs1$lag.BuySell),-99,ifelse(X_lag_bs1$lag.BuySell =='Buy',1,0) )
  #X_lag_bs1 = data.table(X_lag_bs1[,c("CustomerIdx","lag.BuySell_cust")])
  max_d = max(Trade_temp2$TradeDateKey)
  X_date_0_buy <- Trade_temp2[BuySell == 'Buy', .(Max_Date = max(TradeDateKey)), .(CustomerIdx )]
  X_date_0_buy <- X_date_0_buy[, ":="(Last_Date_buy_cust = as.numeric(max_d - Max_Date),
                                      Max_Date = NULL)]
  
  X_date_0_sell <- Trade_temp2[BuySell == 'Sell', .(Max_Date = max(TradeDateKey)), .(CustomerIdx )]
  X_date_0_sell <- X_date_0_sell[, ":="(Last_Date_sell_cust = as.numeric(max_d - Max_Date),
                                        Max_Date = NULL)]
  
  X_date_1 <- Trade_temp2[, .(Max_Date = max(TradeDateKey)), .(CustomerIdx )]
  
  X_date_1 <- X_date_1[, ":="(Last_Date_cust = as.numeric(max_d - Max_Date),
                              Max_Date = NULL)]
  
  
  X_cust_1 <- Reduce(function(x, y) merge(x, y, all=T, by=c("CustomerIdx")),
                     list(all_customer
                          ,X_date_0_buy
                          ,X_date_0_sell
                          ,X_date_1
                     ))
  
  
  X_cust_1[is.na(X_cust_1)] = -999
  X_cust_1
}

Variable_creation_bond = function(Trade_temp,bond_week)
{
  Trade_temp2 = Trade_temp[Trade_temp$CustomerInterest == 1,]
  
  Bond_1 = data.table(Trade_temp)[,.(Number_distinct_customers = length(unique(CustomerIdx))),by = c('IsinIdx')]
  
  Bond_2 = data.table(Trade_temp2)[,.(Num_distinct_c_bs = length(unique(CustomerIdx)),avg_nom = mean(NotionalEUR)),by = c('IsinIdx','BuySell')]
  bond_bs <- as.data.frame(dcast(Trade_temp2, IsinIdx ~ BuySell, value.var = 'CustomerInterest',fun=sum))
  print('Hooraah!!!')
  all_bond = left_join(Bond_1,Bond_2[Bond_2$BuySell=='Buy',],by = 'IsinIdx')
  all_bond = left_join(all_bond,Bond_2[Bond_2$BuySell=='Sell',],by = 'IsinIdx')
  all_bond = left_join(all_bond,bond_bs,by = 'IsinIdx')
  all_bond = all_bond[,-c(3,6)]
  all_bond = left_join(all_bond,bond_week,by = 'IsinIdx')
  all_bond[is.na(all_bond)] = -999
  all_bond
}


####################################################
## Creation of additional customer variables 
##################################################
#customer_precal <- read_csv("customer_precalculated_aggregrate.csv")

##### buy trasactions weekly
Total_master = data.table(Trade)[,.(Interest = sum(CustomerInterest)),by = c('CustomerIdx','week','IsinIdx','BuySell')]
Total_first_interaction = Total_master[,.(min_week = min(week)),by = c("CustomerIdx",'IsinIdx')]

Buy_master = data.table(Trade[Trade$BuySell == 'Buy',])[,.(Interest = sum(CustomerInterest)),by = c('CustomerIdx','week','IsinIdx')]
Buy_m_2 = Buy_master[Buy_master$Interest >0,]
rm(Buy_master)
gc()
#### distinct bonds invested each week
Buy_m_2_number_bonds = Buy_m_2[,.(total_transac = sum(Interest), distinct_n_bonds = .N ),by = c("CustomerIdx",'week')]
#### Number of weeks invested IN 
Buy_m_2_number_of_weeks =  Buy_m_2[,.(min_week = min(week),max_week = max(week), total_weeks = length(unique(week)) ),by = c("CustomerIdx")]
q = "select a.CustomerIdx,a.week,b.IsinIdx from Buy_m_2_number_bonds as a  join Total_first_interaction as b on a.CustomerIdx = b.CustomerIdx and a.week > b.min_week"
temp = sqldf(q) 
temp = data.table(temp)
temp$old = 1
temp = merge(Buy_m_2,temp,by = c("CustomerIdx","week","IsinIdx"), all = T )
temp[is.na(temp)] = 0

Buy1 = dcast(temp[temp$Interest>0,],CustomerIdx~old,value.var='Interest')
Buy1$ratio_old_new = Buy1$`1`/(Buy1$`1`+Buy1$`0`)
colnames(Buy1)[2:3] = c('new_bonds','old_bonds')

rm(temp)
gc()
print('Done Buy')
Sell_master = data.table(Trade[Trade$BuySell == 'Sell',])[,.(Interest = sum(CustomerInterest)),by = c('CustomerIdx','week','IsinIdx')]
Sell_m_2 = Sell_master[Sell_master$Interest >0,]
rm(Sell_master)
gc()
#### distinct bonds invested each week
Sell_m_2_number_bonds = Sell_m_2[,.(total_transac = sum(Interest), distinct_n_bonds = .N ),by = c("CustomerIdx",'week')]
#### Number of weeks invested IN 
Sell_m_2_number_of_weeks =  Sell_m_2[,.(min_week = min(week),max_week = max(week), total_weeks = length(unique(week)) ),by = c("CustomerIdx")]
q = "select a.CustomerIdx,a.week,b.IsinIdx from Sell_m_2_number_bonds as a  join Total_first_interaction as b on a.CustomerIdx = b.CustomerIdx and a.week > b.min_week"
temp = sqldf(q) 
temp = data.table(temp)
temp$old = 1
temp = merge(Sell_m_2,temp,by = c("CustomerIdx","week","IsinIdx"), all = T )
temp[is.na(temp)] = 0

Sell1 = dcast(temp[temp$Interest>0,],CustomerIdx~old,value.var='Interest')
Sell1$ratio_old_new = Sell1$`1`/(Sell1$`1`+Sell1$`0`)
colnames(Sell1)[2:3] = c('new_bonds','old_bonds')

rm(temp,Total_master,Total_first_interaction)

Trade2 = Trade[Trade$CustomerInterest == 1,]
total_minweek = data.table(Trade)[,.(min_week = min(week_numeric)),by = c('CustomerIdx')]
nonholding_minweek = data.table(Trade2)[,.(min_week_non_holding = min(week_numeric),distinct_weeks = length(unique(week_numeric)) ),by = c('CustomerIdx')]
nonholding_buysell_distinct_weeks = data.table(Trade2)[,.( distinct_weeks_bs = length(unique(week_numeric)) ),by = c('CustomerIdx','BuySell')]
week_related = left_join(total_minweek,nonholding_minweek,by =  c('CustomerIdx') )
week_related = left_join(week_related,nonholding_buysell_distinct_weeks[nonholding_buysell_distinct_weeks$BuySell=='Buy',],by =  c('CustomerIdx') )
week_related = left_join(week_related,nonholding_buysell_distinct_weeks[nonholding_buysell_distinct_weeks$BuySell=='Sell',],by =  c('CustomerIdx') )

week_related$Lag_week_121 = (121 - week_related$min_week)/(week_related$distinct_weeks )

week_related$Lag_week_buy_121 = (121 - week_related$min_week)/(week_related$distinct_weeks_bs.x )

week_related$Lag_week_sell_121 = (121 - week_related$min_week)/(week_related$distinct_weeks_bs.y )


cust_week1 <- week_related[,c(1,9,10,11)]
cust_week1[is.na(cust_week1)] <- 9999

customer_precal <- left_join(cust_week1,Buy1[,c(1,4)],by = "CustomerIdx")
customer_precal <- left_join(customer_precal,Sell1[,c(1,4)],by = "CustomerIdx")
customer_precal[is.na(customer_precal)]<- 0
### write to a file for  saving time in future ###
#fwrite(customer_precal,"customer_precalculated_aggregrate.csv")

###############################################
#### Creation of additional bond variables ####
###############################################
#bond_week_2 <- read_csv("bond_week_1.csv")

total_minweek = data.table(Trade)[,.(min_week = min(week_numeric)),by = c('IsinIdx')]
nonholding_minweek = data.table(Trade2)[,.(min_week_non_holding = min(week_numeric),distinct_weeks = length(unique(week_numeric)) ),by = c('IsinIdx')]
nonholding_buysell_distinct_weeks = data.table(Trade2)[,.( distinct_weeks_bs = length(unique(week_numeric)) ),by = c('IsinIdx','BuySell')]
week_related = left_join(total_minweek,nonholding_minweek,by =  c('IsinIdx') )
week_related = left_join(week_related,nonholding_buysell_distinct_weeks[nonholding_buysell_distinct_weeks$BuySell=='Buy',],by =  c('IsinIdx') )
week_related = left_join(week_related,nonholding_buysell_distinct_weeks[nonholding_buysell_distinct_weeks$BuySell=='Sell',],by =  c('IsinIdx') )

week_related$Lag_week_121 = (121 - week_related$min_week)/(week_related$distinct_weeks )

week_related$Lag_week_buy_121 = (121 - week_related$min_week)/(week_related$distinct_weeks_bs.x )

week_related$Lag_week_sell_121 = (121 - week_related$min_week)/(week_related$distinct_weeks_bs.y )
bond_week_1 <- week_related[,c(1,9,10,11)]
bond_week_1[is.na(bond_week_1)] <- 9999
#fwrite(bond_week_1,"bond_week_1.csv")


###################################
##### Market related features #####
###################################


Isin=read_csv("Isin.csv")

#### Function to calculate fixed coupon rate
Fixed_coupon = function(cp,y,T)
{
  coupon = (y/2)*(cp - 100/(1+(y/2))^T )/ (1 - (1/(1+(y/2))^T ))
  
  coupon
}
market <- read_csv("C:/Users/Yash/Desktop/dsg/Market.csv")
market$DateKey = as.Date(as.character(market$DateKey), format = '%Y%m%d')
market$week = as.numeric(paste(format(market$DateKey,'%y'),format(market$DateKey,"%V"),sep = ''))
market$week = replace(market$week,market$week==1653,1553)
market$week_numeric = as.numeric(as.factor(market$week))

market2 = left_join(market,Isin[,c(1,3,17)],by = c("IsinIdx") )
market2$ActualMaturityDateKey = as.Date(as.character(market2$ActualMaturityDateKey), format = '%Y%m%d')
market2$Timetomature = as.numeric(market2$ActualMaturityDateKey-market2$DateKey)
market2$Timetomature = replace(market2$Timetomature,market2$Timetomature<0,0)
market2$semi_annual_periods = round(market2$Timetomature/182)
market2$near_coupon_date = market2$Timetomature/182 - market2$semi_annual_periods
market2$Yield2 = replace(market2$Yield,(!is.na(market2$Yield))&(market2$Yield >30),999 )
market2$Yield2 = replace(market2$Yield,(!is.na(market2$Yield))&(market2$Yield < -30),-999 )
market2$Coupon_amount = Fixed_coupon(market2$Price,market2$Yield2/100,market2$semi_annual_periods)

colnames(market2)[2] = "TradeDateKey"

market3 = data.table(market2)[,.(TradeDateKey = max(TradeDateKey)
                                 ,ActualMaturityDateKey = max(ActualMaturityDateKey)
                                 ,CouponType = max(CouponType)
                                 ,Price = mean(Price)
                                 ,Yield = mean(Yield)
                                 #,Yield2= mean(Yield2)
                                 ,sd_Yield = var(Yield)
                                 ,sd_z = var(ZSpread)
                                 , Coupon_amount= mean(Coupon_amount)
                                 ,ZSpread = mean(ZSpread)
                                 , Timetomature = max(Timetomature)
                                 ,semi_annual_periods = max(semi_annual_periods)
                                 , near_coupon_date = mean(near_coupon_date) )
                              ,by = c('IsinIdx','week','week_numeric') ]

market3$Coupon_amount = replace(market3$Coupon_amount,(market3$Coupon_amount >30),999 )
market3$Coupon_amount = replace(market3$Coupon_amount,(-market3$Coupon_amount > 30),-999 )
market3$Coupon_amount = replace(market3$Coupon_amount,is.na(market3$Coupon_amount),200 )
market3[is.na(market3)] <- 0


market3$perc_yield=((market3$Yield-lag(market3$Yield))/lag(market3$Yield))*100
market3$perc_price=((market3$Price-lag(market3$Price))/lag(market3$Price))*100
market3$perc_coupon=((market3$Coupon_amount-lag(market3$Coupon_amount))/lag(market3$Coupon_amount))*100
market3[is.na(market3)] <- 0
market3$perc_yield <- replace(market3$perc_yield,market3$perc_yield>999,999)
market3$perc_yield <- replace(market3$perc_yield,-market3$perc_yield>999,-999)
market3$perc_coupon <- replace(market3$perc_coupon,market3$perc_coupon>999,999)
market3$perc_coupon <- replace(market3$perc_coupon,-market3$perc_coupon>999,-999)
market3$duration = (market3$perc_price + 0.001)/(market3$perc_yield + 0.001)
market3 = market3[,-c(4,5,6)]

fwrite(market3,'Market_related_new.csv')

fwrite(market3[market3$week_numeric==121,],'Market_related_test.csv')


################################
################################

################################
##### MAIN FUNCTION ############
################################

main_funct = function(Tr,w,customer_precal,bond_week_1)
{
  customer_add_vars = Variable_creation_customer(Tr,customer_precal)
  bond_add_vars = Variable_creation_bond(Tr,bond_week_1)
  t_h = Variable_creation_func_holding(Tr,w)
  print("done...1")
  t_nh= Variable_creation_func1(Tr,w)
  print("done...2")
  t_both = Variable_creation_func_both(Tr,w)
  print("done...3")
  t_both = left_join(t_both,t_nh,by= c("CustomerIdx" , "IsinIdx"))
  t_both = left_join(t_both,t_h,by= c("CustomerIdx" , "IsinIdx"))
  print("done...4")
  t_both = left_join(t_both,customer_add_vars,by= c("CustomerIdx"))
  t_both = left_join(t_both,bond_add_vars,by= c("IsinIdx"))
  t_both[is.na(t_both)] = -999
  t_both = data.table(t_both)
  t_both$cust.bond = as.character(paste(t_both$CustomerIdx,t_both$IsinIdx,sep ='|'))
  
  t_both
}


Traget_creation = function(Train,Tr,w,buysell)
{
  data <- Tr[Tr$CustomerInterest>0,]
  X_val_1 <- data[data$BuySell==buysell & data$week_numeric == w,]
  target <- ifelse(Train$cust.bond %in% unique(X_val_1$cust.bond), 1, 0)
  
}


cummulative_func = function(df)
{
  df$Buy_2 = df$Buy_1 + df$Buy_2
  df$Buy_3 = df$Buy_2 + df$Buy_3
  df$Buy_4 = df$Buy_3 + df$Buy_4
  df$Buy_5 = df$Buy_5 + df$Buy_5
  df$Sell_2 = df$Sell_1 + df$Sell_2
  df$Sell_3 = df$Sell_2 + df$Sell_3
  df$Sell_4 = df$Sell_3 + df$Sell_4
  df$Sell_5 = df$Sell_5 + df$Sell_5
  df
}


########################################

### Week 121 data ###

tic = Sys.time()
Train_121 = main_funct(Trade,121,customer_precal,bond_week_1)
Train_121$Target_Buy =  Traget_creation(Train_121,Trade,121,'Buy')
Train_121$Target_Sell =  Traget_creation(Train_121,Trade,121,'Sell')
tac =  Sys.time()
print(tac-tic)

Train_121 = cummulative_func(Train_121)
fwrite(Train_121,'DSG_Train_final_week_121_2.csv')

rm(Train_121)
gc()

### Week 120 data ###

tic = Sys.time()
Train_120 = main_funct(Trade,120,customer_precal,bond_week_1)
Train_120$Target_Buy =  Traget_creation(Train_120,Trade,120,'Buy')
Train_120$Target_Sell =  Traget_creation(Train_120,Trade,120,'Sell')
Train_120 = cummulative_func(Train_120)
tac =  Sys.time()
print(tac-tic)

Train_120 = cummulative_func(Train_120)
fwrite(Train_120,'DSG_Train_final_week_120_2.csv')

rm(Train_120)
gc()


### Week 119 data ###

tic = Sys.time()
Train_119 = main_funct(Trade,119,customer_precal,bond_week_1)
Train_119$Target_Buy =  Traget_creation(Train_119,Trade,119,'Buy')
Train_119$Target_Sell =  Traget_creation(Train_119,Trade,119,'Sell')

tac =  Sys.time()

print(tac-tic)

Train_119 = cummulative_func(Train_119)
fwrite(Train_119,'DSG_Train_final_week_119_2.csv')
rm(Train_119)


### Week 118 data ###

tic = Sys.time()
Train_118 = main_funct(Trade,118,customer_precal,bond_week_1)
Train_118$Target_Buy =  Traget_creation(Train_118,Trade,118,'Buy')
Train_118$Target_Sell =  Traget_creation(Train_118,Trade,118,'Sell')

tac =  Sys.time()

print(tac-tic)
Train_118 = cummulative_func(Train_118)
fwrite(Train_118,'DSG_Train_final_week_118_2.csv')
rm(Train_118)
gc()

#### Creation of Test Dataset ########
tic = Sys.time()
Test = main_funct(Trade,122,customer_precal,bond_week_1)
tac =  Sys.time()

Test_final = left_join(Challenge_20180423[,-2],Test,by= c("CustomerIdx","IsinIdx"))
Test_final$BuySell = ifelse(Test_final$BuySell == 'Buy',1,0)

Test_final = cummulative_func(Test_final)

fwrite(Test_final,'Test_final.csv')

#################################################################
