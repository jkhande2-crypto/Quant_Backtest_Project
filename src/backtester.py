#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 18:14:21 2026

@author: jaykhandelwal
"""

import numpy as np
import pandas as pd

def backtest(data, transaction_cost=0.001):
    df = data.copy()

    # Position_Change detects when we enter or exit a trade
    # diff() gives difference between today's signal and yesterday's signal
    # abs() because we count both entries (0->1) and exits (1->0)
    
    df['Position_Change']= df['Signal'].diff().abs().fillna(0) 
    
    # Cost is only charged on days when we actually trade
    df['Transaction_Cost']=df['Position_Change']* transaction_cost
    
    # Net return = strategy return minus any trading cost that day
    df['Net_Strategy_return'] = df['Strategy_Returns']- df['Transaction_Cost']
    
    
    # Cumulative product builds the equity curve
   
    
    df['Cumulative_Market']= (1+df['Returns']).cumprod()
    df['Cumulative_Strategy']= (1+df['Net_Strategy_return']).cumprod()
    
    return df

#.cumprod() - means cumulative product 

#(1+returns).cumprod() - builds our equity curve. That's how we calculate cumulative strategy growth.

#.diff() - calculates the difference between current row and previous row. 
# if signal is:
#0
#0
#1
#1
#0

#then .diff() gives:
#NaN
#0
#1
#0
#-1

#If strategy cumulative return > market cumulative return, we beat the market.
#If strategy cumulative return < market cumulative return, our strategy underperformed
