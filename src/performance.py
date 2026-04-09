#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 18:14:29 2026

@author: jaykhandelwal
"""

import numpy as np
import pandas as pd
 
def compute_drawdown(cum_curve): 
   
    # Drawdown tells us how far we are from the previous peak at any point
    # peak = highest value seen so far at each date
    
    
    peak = cum_curve.cummax()  
   
    # negative number — how far below peak we are
    #computing drawdown as a %
    
    dd = (cum_curve - peak)/peak   
    return dd

def summarize_performance(data, returns_col, cum_col):
    # Summary metrics for a returns series + cumulative curve.
    
    r = data[returns_col].dropna()       # removes missing values from return column (NaN values)
    
    ann_return = r.mean() * 252          # average daily return * 252 trading days
    
    ann_vol = r.std() * np.sqrt(252)     # daily std * sqrt(252) to annualize
    
    sharpe = ann_return/ann_vol if ann_vol !=0 else np.nan   
    
    dd = compute_drawdown(data[cum_col])   
    max_dd = dd.min()  # drawdown values are negative so .min() gives worst drawdown
    
    
    # win rate = fraction of days with a positive return
    # r > 0 gives True/False, .mean() of booleans = fraction of True values
    win_rate = (r > 0).mean()    
    
    
    # CAGR using start and end values of the cumulative curve
    
    n_days= len(r)   
    start_val = data[cum_col].iloc[0]
    end_val = data[cum_col].iloc[-1]
    
    cagr = ((end_val / start_val) **(252 / n_days) - 1) if n_days > 0 else np.nan   # True & False condition. True if > 0 else False. Avoids error
    
    # Sortino ratio — like Sharpe but only penalizes downside volatility
    
    downside_vol = r[r < 0].std() * np.sqrt(252)
    
    sortino = (ann_return / downside_vol) if downside_vol > 0 else np.nan
   
    
    return{
        "CAGR" : round(float(cagr),4),
        "Annual_Return" : round(float(ann_return),4),
        "Annual_Volatility" : round(float(ann_vol),4),
        "Sharpe" : round(float(sharpe),4),
        "Sortino" : round(float(sortino),4),
        "Max_Drawdown" : round(float(max_dd),4),
        "Win_Rate_Daily" : round(float(win_rate),4)
        }

def print_metrics(title, strat_metrics, mkt_metrics):
    
    def pct(x):
        return f"{x * 100:.2f}%"
    
    print(f"\n====== {title} ======")
    print(f"  Strategy | CAGR: {pct(strat_metrics['CAGR'])} | Sharpe: {strat_metrics['Sharpe']:.4f} | Sortino: {strat_metrics['Sortino']:.4f} | Max DD: {pct(strat_metrics['Max_Drawdown'])}")
    print(f"  Market   | CAGR: {pct(mkt_metrics['CAGR'])}   | Sharpe: {mkt_metrics['Sharpe']:.4f} | Sortino: {mkt_metrics['Sortino']:.4f} | Max DD: {pct(mkt_metrics['Max_Drawdown'])}")
    
    
    
def period_report(data, start, end, label = "Period"):  # defined a function named perioed_report that will give us a small report with start to end data and will label it as "PERIOD"
    # Prints a mini report for a specific crisis period 
    
    sub = data.loc[start : end].copy()  
    #  data.loc[start : end] - cuts the data / DataFrame by date ( take it as we are fitering our data by indexing where we have given rows - start to end date) & takes the data of that menioned perioed.
    # .copy()  - it makes a new data / DataFrame so that we dont change the orginal data
    
    if len(sub) < 5:   # if the length of that period is < 5, print 
        print(f"\n[{label}] Not enough data.")
        return
    
   # Rescale so both curves start at 1.0 at the beginning of this period
   # This lets us measure performance purely within the crisis window
    
    mkt_curve = sub["Cumulative_Market"] / sub["Cumulative_Market"].iloc[0]   # current / previos value. .iloc[0] - integer indexing starting from row 0.
    strat_curve = sub["Cumulative_Strategy"] / sub["Cumulative_Strategy"].iloc[0]
    # its a trick for a crisis period. it forces both curves to start at 1.0  at the beginning of the period
    # and then we compare how much each grew / fell during that period.
    

    mkt_dd = compute_drawdown(mkt_curve).min()    
    strat_dd = compute_drawdown(strat_curve).min()
    
    mkt_ret = mkt_curve.iloc[-1] - 1     
    strat_ret = strat_curve.iloc[-1] - 1 
    
    
    # Exposure = avg of signal column = fraction of days we were invested
    # Example - exposure 0.65 means we were long 65% of the days.
    exposure = sub["Signal"].mean()  
    
    
    
    # Count entries: days where signal went from 0 to 1
    entries = int((sub["Signal"].diff() == 1).sum())
    
    # no. of Long entries
    # .diff() - detects changes day-to-day
    # ==1 - counts only transition from 0 -> 1 (entering a long position)
    # .sum() -  counts how many such entries happened.
    
    
    print(f"\n====== {label} ({start} to {end}) ======")
    print(f"  Market   | Return: {mkt_ret:.2%}   | Max DD: {mkt_dd:.2%}")
    print(f"  Strategy | Return: {strat_ret:.2%} | Max DD: {strat_dd:.2%}")
    print(f"  Exposure (avg signal): {exposure:.2%}")
    print(f"  Number of Entries: {entries}")


