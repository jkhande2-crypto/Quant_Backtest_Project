#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 18:14:03 2026

@author: jaykhandelwal
"""

import pandas as pd

def moving_average_strategy (data, short_window=50, long_window=200):
    
    df=data.copy()
    
    df['MA_Short']=df['Close'].rolling(window=short_window).mean()
    
    df['MA_Long']=df['Close'].rolling(window=long_window).mean()
    
    # Signal = 1 when short MA is above long MA, else 0
    # This is the raw signal for today
    df["Signal"] = (df["MA_Short"] > df["MA_Long"]).astype(int)
    
    #Prevent Look-Ahead Bias: trade tomorrow basedon today's signal
    df['Strategy_Returns']=df['Signal'].shift(1).fillna(0) * df['Returns'] 
    
    return df

    # data['Signal'].shift(1) - Prevents look-ahead bias. 
    # We use .shift(1) to ensure that today’s trading decision is based only on information available at the close of the previous day. 
    # Without shifting, the strategy would unrealistically use today’s signal to capture today’s return, introducing look-ahead bias and overstating performance.
    # it moves data down by 1 row. 
    
#if signal is :
#Day 1: 0
#Day 2: 1
#Day 3: 1

# Signal.shift(1) will make the signal: 
#Day 1: NaN
#Day 2: 0
#Day 3: 1
#Day 4: 1
# it because we cant trade using today's signal on today's return. we only know today's signal after market closes. So, w/o using ".shit(1), we are CHEATING. That's called "LOOK-AHEAD BIAS".


 # Look-ahead bias is a, research error where a model or strategy is built using information that wasn't available at the time of the decision. 
 # Using future data to predict or simulate past events.  
   
# Signal=1 Long Position
# Signal=0  No Position
# When 50 MA > 200 MA, we long the position; otherwise no
# MA_Short is the average of the last 50 closing price. Reacts faster to recent price chages.
# MA_Long is the average of the last 200 closing prices. It reacts slower to recent price changes and shows long-term trend.

# Our rule was:
    # If MA_Short > MA_Long ; Signal = 1 meaning buy ( Signal = 0 meaning no position)
    
# .astype(int) - used withing the pandas / numpy libraries to cast a elements of aa Series, DataFrame or Array to an integer data type.
    