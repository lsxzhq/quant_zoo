import math
import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from gplearn.genetic import SymbolicTransformer
from gplearn.fitness import make_fitness
import gplearn as gp
import time
import tushare as ts 
import empyrical
import talib as ta 

pro = ts.pro_api()
hs300 = pro.index_daily(ts_code='000300.SH').sort_values('trade_date').reset_index(drop=True).query('trade_date>="2010-01-01"').reset_index(drop=True)
L = ['open','close','high','low','vol']
#train_x = hs300[['open','close','high','low','vol']].iloc[:1950]
#train_y = hs300['pct_chg'].iloc[:1951].shift(-1).dropna()
train_x = hs300[['open','close','high','low','vol']].copy()
train_y = hs300['pct_chg'].shift(-1).fillna(0)
data_x = hs300[['open','close','high','low','vol']].copy()
for i in [1,3,5,7,10,15,20,25]:
    train_x.loc[:,str(i)] = i
    data_x.loc[:,str(i)] = i
    L.append(str(i))
#%%
def _ts_rank(data,n):
    
    with np.errstate(divide='ignore', invalid='ignore'):

        try:
            if  (n[0] == n[1] and n[1] ==n[2]) and n[3]==n[2]:        
                window = n[0]
                value = np.array(pd.Series(data.flatten()).rolling(window).apply(_rolling_rank).tolist())
                value = np.nan_to_num(value)

                return value
            else:
                return data
        except:
            return data
        
def _ts_ma(data,n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if  (n[0] == n[1] and n[1] ==n[2]) and n[3]==n[2]:  
                window = n[0]
                ma = np.array(pd.Series(data.flatten()).rolling(window).mean().tolist())
                ma = np.nan_to_num(ma)
                return ma 
            else:
                return data
        except:
            return data

def _delay(data,n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] ==n[2]:  
                window = n[0]
                data = np.array(pd.Series(data.flatten()).shift(window).tolist())
                data = np.nan_to_num(data)
                return data
            else:
                return data
        except:
            return data
def _stddev(data,n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] ==n[2] and n[2]==n[3]:  
                window = n[0]
                data = np.array(pd.Series(data.flatten()).shift(window).std().tolist())
                data = np.nan_to_num(data)
                return data
            else:
                return data
        except:
            return data
def _delta(data,n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] ==n[2] and n[3]==n[2]:  
                window = n[0]
                data = np.array((pd.Series(data.flatten()) - pd.Series(data.flatten()).shift(window)).tolist())
                data = np.nan_to_num(data)
                return data
            else:
                return data
        except:
            return data
def _ts_sum(data,n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if (n[0] == n[1] and n[1] ==n[2]) and n[3]==n[2]:  
                window = n[0]
                data = np.array(pd.Series(data.flatten()).rolling(window).sum().values)
                data = np.nan_to_num(data)
                return data
            else:
                return data
        except:
            return data        
        
def _ts_max(data,n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if  (n[0] == n[1] and n[1] ==n[2]) and n[3]==n[2]:  
                window = n[0]
                data = np.array(pd.Series(data.flatten()).rolling(window).max().tolist())
                data = np.nan_to_num(data)
                return data
            else:
                return data
        except:
            return data   
        
def _ts_min(data,n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if (n[0] == n[1] and n[1] ==n[2]) and n[3]==n[2]:
                window = n[0]
                data = np.array(pd.Series(data.flatten()).rolling(window).min().tolist())
                data = np.nan_to_num(data)
                return data
            else:
                return data
        except:
            return data   
def _ts_prod(data,n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if (n[0] == n[1] and n[1] ==n[2]) and n[3]==n[2]:
                if n[0]<=25:
                    window = n[0]
                    data = np.array((pd.Series(data.flatten()).rolling(window)).cumprod().values)
                    data = np.nan_to_num(data)
                    return data
                else:
                    return data
            else:
                return data
        except:
            return data   
def _clear_by_cond(a,b,c):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            a1 = pd.Series(a.flatten())
            b1 = pd.Serise(b.flatten())
            c1 = pd.Series(c.flatten())
            return np.where(a1<b1,0,c1)
        except:
            return a 
def _ewma(data,n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if (n[0] == n[1] and n[1] ==n[2]) and n[3]==n[2]:
                window = n[0]
                data = np.array(pd.Series(data.flatten()).ewm(span=20,min_periods=1).mean())
                data = np.nan_to_num(data)
                return data
            else:
                return data
        except:
            return data
def _mean3(a,b,c):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            return (a+b+c)/3
        except:
            return a
def _mean2(a,b):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            return (a+b)/2
        except:
            return a        
def _corr(a,b,n):
    with np.errstate(divide='ignore', invalid='ignore'):
       try:
           if n[0] == n[1] and n[1] ==n[2] and n[3]==n[2]:
               window  = n[0]
               x1 = pd.Series(a.flatten())
               x2 = pd.Series(b.flatten())
               df = pd.concat([x1,x2],axis=1)
               temp = pd.Series(dtype='float64')
               for i in range(len(df)):
                   if i<=window-2:
                       temp[str(i)] = np.nan
                   else:
                       df2 = df.iloc[i-window+1:i,:]
                       temp[str(i)] = df2.corr('spearman').iloc[1,0]
               return np.nan_to_num(temp)
           else:
               return a
           
       except:
           return a
def _ta_KAMA(data,n):
    with np.errstate(divide='ignore', invalid='ignore'):
       try:
           if n[0] == n[1] and n[1] ==n[2] and n[3]==n[2]:
               window=n[0]
               data = np.array(ta.KAMA(pd.Series(data.flatten()),timeperiod=window).tolist())
               data = np.nan_to_num(data)
               return data
           else:
               return data
       except:
           return data
def _ta_DEMA(data,n):
    with np.errstate(divide='ignore', invalid='ignore'):
       try:
           if n[0] == n[1] and n[1] ==n[2] and n[3]==n[2]:
               window=n[0]
               data = np.array(ta.DEMA(pd.Series(data.flatten()),timeperiod=window).tolist())
               data = np.nan_to_num(data)
               return data
           else:
               return data
       except:
           return data
def _ta_ht_dcphase(data):
    with np.errstate(divide='ignore', invalid='ignore'):
       try:
               data = np.array(ta.HT_DCPHASE(pd.Series(data.flatten())).tolist())
               data = np.nan_to_num(data)
               return data
       except:
           return data
def _beta(a,b,n):
    with np.errstate(divide='ignore', invalid='ignore'):
       try:
           if n[0] == n[1] and n[1] ==n[2] and n[3]==n[2]:
               window  = n[0]
               x1 = pd.Series(a.flatten())
               x2 = pd.Series(b.flatten())
               df = pd.concat([x1,x2],axis=1)
               temp = pd.Series(dtype='float64')
               for i in range(len(df)):
                   if i<=window-2:
                       temp[str(i)] = np.nan
                   else:
                       df2 = df.iloc[i-window+1:i,:]
                       temp[str(i)] = empyrical.beta(df.iloc[:,0],df.iloc[:,1])
               return np.nan_to_num(temp)
           else:
               return a
       except:
           return a
def _cov(a,b,n):
    with np.errstate(divide='ignore', invalid='ignore'):
       try:
           if n[0] == n[1] and n[1] ==n[2] and n[3]==n[2]:
               window  = n[0]
               x1 = pd.Series(a.flatten())
               x2 = pd.Series(b.flatten())
               df = pd.concat([x1,x2],axis=1)
               temp = pd.Series(dtype='float64')
               for i in range(len(df)):
                   if i<=window-2:
                       temp[str(i)] = np.nan
                   else:
                       df2 = df.iloc[i-window+1:i,:]
                       temp[str(i)] = df2.cov().iloc[1,0]
               return np.nan_to_num(temp)
           else:
               return a
           
       except:
           return a       
cov = gp.functions.make_function(function =_cov, name='cov', arity = 3)   
beta = gp.functions.make_function(function =_beta, name='beta', arity = 3)
ta_ht_dcphase = gp.functions.make_function(function =_ta_ht_dcphase, name='ta_ht_dcphase', arity = 1)    
ta_DEMA =  gp.functions.make_function(function =_ta_DEMA, name='ta_DEMA', arity = 2)
ta_KAMA =  gp.functions.make_function(function =_ta_KAMA, name='ta_KAMA', arity = 2)
corr = gp.functions.make_function(function =_corr, name='corr', arity = 3)
mean3 = gp.functions.make_function(function =_mean3,name = 'mean3',arity = 3)      
mean2 = gp.functions.make_function(function =_mean2,name = 'mean2',arity = 2)      
ewma = gp.functions.make_function(function =_ewma,name = 'ewma',arity = 2)
clear_by_cond = gp.functions.make_function(function =_clear_by_cond,name = 'clear_by_cond',arity = 3)            
ts_prod = gp.functions.make_function(function =_ts_prod,name = 'ts_prod',arity = 2)           
ts_max = gp.functions.make_function(function =_ts_max,name = 'ts_max',arity = 2)           
ts_min = gp.functions.make_function(function =_ts_min,name = 'ts_min',arity = 2)       
ts_sum = gp.functions.make_function(function =_ts_sum,name = 'ts_sum',arity = 2)
ts_rank = gp.functions.make_function(function =_ts_rank,name = 'ts_rank',arity = 2)
ts_ma = gp.functions.make_function(function =_ts_ma,name = 'ts_ma',arity = 2)
delay = gp.functions.make_function(function =_delay,name = 'delay',arity =2)
stddev = gp.functions.make_function(function =_stddev,name = 'stddev',arity =2)
delta = gp.functions.make_function(function =_delta,name = 'delta',arity =2)


user_function = [ts_rank,ts_ma,delta,stddev,delay,ts_max,ts_min]#clear_by_cond,ewma,corr,ta_KAMA,ta_DEMA,ta_ht_dcphase,beta,mean2,mean3,ts_sum,ts_prod,cov]
#ts_sum,ts_prod
def _cum(y, y_pred, w):
    
    y_pre = pd.DataFrame({'f':y_pred.flatten()}).fillna(0)
    y_pre['80q'] = y_pre['f'].rolling(60).quantile(0.2)
    y_pre['20q'] = y_pre['f'].rolling(60).quantile(0.8)
    basic_signal = []
    for row in range(len(y_pre)):
        if y_pre['f'][row] >= y_pre['80q'][row]:
            basic_signal.append(1)
        else:
            if row != 0:
                if basic_signal[-1] and y_pre['f'][row] > y_pre['20q'][row]:
                    basic_signal.append(1)
                else:
                    
                    basic_signal.append(0)
            else:
                basic_signal.append(0)
    y_pre['positions'] = basic_signal
    y_pre['ret'] = y_pre['positions']*pd.Series(y)*0.01
    total = (((y_pre['positions'][1:]!=y_pre['positions'].shift(1).dropna()).sum())/252 +1)
    buy_day = (y_pre['positions']).sum()/(total*252)
    total_day = (y_pre['positions']).sum() + 1
    win = np.sum(np.where(y_pre['ret'] > 0, 1, 0))
    win_ratio = win/total_day
    #print((1+y_pre['ret']).cumprod().iloc[-1])
    try:
        #my_metric = (1+y_pre['ret']).cumprod().iloc[-1]
        my_metric_new = empyrical.calmar_ratio(y_pre['ret'])*np.sqrt(buy_day)*win_ratio*np.sqrt(total)*my_metric
        
    except:
        my_metric_new = -1
    
    del y 
    del y_pre
    del y_pred
    del basic_signal
    
    return  np.mean(my_metric)

cum = gp.fitness.make_fitness(function=_cum,greater_is_better=True)

init_function = ['add', 'sub', 'mul', 'div', 'sqrt', 'abs','log','max','min','neg']


gp1 = SymbolicTransformer(
    generations=2, #整数，可选(默认值=20)要进化的代数
    population_size=1000,# 整数，可选(默认值=1000)，每一代群体中的公式数量
    hall_of_fame=500,
    n_components=10,#最终生成的因子数量
    function_set=init_function+user_function ,
    parsimony_coefficient=0.0005,
    p_crossover=0.9,
    p_point_replace=0.4,
    max_samples=1,
    verbose=1,
    const_range = None,
    feature_names=list('$'+n for n in L),
    random_state=42,  # 随机数种子
    init_depth=(2,6),
    metric=cum,
    n_jobs=12,stopping_criteria=np.inf,warm_start=True
)

gp1.fit(train_x,train_y)
print(gp1)
#%%

res = pd.DataFrame(gp1.transform(data_x))

            
                   
    