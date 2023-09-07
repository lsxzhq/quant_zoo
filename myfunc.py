def _cum(y, y_pred, w):
    
    y_pre = pd.DataFrame({'f':y_pred.flatten()},index=global_index).fillna(0)
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