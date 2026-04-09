#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 04:52:31 2026

@author: jaykhandelwal
"""

# This answers the question: How does the strategy actually behaves over time?
# How often is it invested? How long does it hold positions?
# Are there specific months or years where it does well/badly?

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def compute_exposure_stats(bt_df, label = "Full Period"):
    """
    Exposure = fraction of days wehere signal = 1 (where we hold long position)
    This tells us how much time the strategy actually spends in the market.
    A trend-following strategy should be Out of the Market during the bad periods.
    
    """
    
    signal = bt_df["Signal"].dropna()
    
    # Overall time in the market
    pct_invested = signal.mean()
    
    # Avg Holding Period ( consecutive 1's = 1 trade). we do this by fnding where the signal changes and measures gaps  
    trade_start = signal.diff() == 1
    trade_end = signal.diff() == -1
    
    hold_lengths = []
    current_hold = 0
    
    for val in signal:
        if val == 1:
            current_hold += 1
        else:
            if current_hold > 0:
                hold_lengths.append(current_hold)
            current_hold = 0
            
    if current_hold > 0:
        hold_lengths.append(current_hold)
        
        
    avg_hold = np.mean(hold_lengths) if hold_lengths else 0
    max_hold = np.max(hold_lengths) if hold_lengths else 0
    min_hold = np.min(hold_lengths) if hold_lengths else 0
    n_trades = len(hold_lengths)
    
    print(f"\n ------- Exposure Analysis: {label} --------")
    
    print(f"\n Time Invested (Long): {pct_invested: .2%}")
    
    print(f"\n TIme Out of the Market: { 1- pct_invested: .2%}")
    
    print(f"\n Number of Trades: {n_trades}")
    
    print(f"\n Avg Holding Period: {avg_hold: .1f} days ({avg_hold / 21: .1f} months)")
    
    print(f"\n Longest Hold: {max_hold} days ({ max_hold / 21: .1f} months)")
    
    print(f"\n Shortest Hold: {min_hold} days ({min_hold / 21: .1f} months)")
    
    return { 
        "Pct_Invested" : round(pct_invested, 4),
        "N_Trades" : n_trades,
        "Avg Hold Days" : round(avg_hold, 1),
        "Max Hold Days" : max_hold,
        "Min Hold Days" : min_hold,
        }

def plot_annual_returns (bt_df, label = "Strategy"):
    """
    Bar Chart comparing strat vs mkt ret for each calendar year.
    This will tell us which year(s) the strategy helped.
    """
    
    bt_df = bt_df.copy()
    bt_df ["Year"] = bt_df.index.year
    
    # Annual Returns = (1+r).cumprod() for each year
    annual = bt_df.groupby("Year").apply(
        lambda g: pd.Series({
            "Strategy" : (1 + g["Net_Strategy_return"]).prod() - 1,
            "Market" : (1 + g["Returns"]).prod() - 1,
            })
        )

    fig, ax = plt.subplots(figsize = (16,5))

    x = np.arange(len(annual))
    width = 0.38

    bars1 = ax.bar( x - width/2, annual["Strategy"], width, label = "Strategy", color = "steelblue", alpha = 0.85)
    bars2 = ax.bar (x + width/2, annual["Market"], width, label = "Market (B&H) ", color ="lightcoral", alpha = 0.85)

    ax.axhline ( 0, color = "black", linewidth = 0.8)
    ax.set_xticks(x)
    ax.set_xticklabels( annual.index, rotation = 45, fontsize = 9)
    ax.set_ylabel( "Annual Return")
    ax.set_title(f"Annual Return : Strategy vs Market ({label})", fontsize = 13)
    ax.yaxis.set_major_formatter( plt.FuncFormatter( lambda y, _: f"{y : .0%}"))
    ax.legend()
    ax.grid( True, alpha = 0.2, axis = "y")

    plt.tight_layout()
    plt.show()

def plot_monthly_heatmap( bt_df, col ="Net_Strategy_return", label = "Strategy"):
    """
    Monthly return heatmap : rows = years, col = months
    Green = positive, Red = Negative
    """
    
    bt_df = bt_df.copy()
    monthly = bt_df[col].resample("ME").apply( lambda r: (1+r).prod()-1)
    
    # Building a pivot : rows (years), columns (months number)
    df_m = monthly.to_frame("Return")
    df_m ["Year"] = df_m.index.year 
    df_m["Month"] = df_m.index.month
    
    pivot = df_m.pivot( index = "Year", columns = "Month", values = "Return")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul","Aug","Sept", "Oct", "Nov", "Dec"]
    pivot.columns = [month_names[m-1] for m in pivot.columns]
    
    fig, ax = plt.subplots( figsize = (16, max (6, len(pivot) * 0.45)))
    
    im = ax.imshow( pivot.values, cmap = "RdYlGn", aspect = "auto")
    
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names, fontsize = 9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize = 9)
    
    # For Return % in each cell
    for i in range(len(pivot.index)):
        for j in range(12):
            val = pivot.values[i,j]
            if not np.isnan(val):
                ax.text( j, i, f"{val:.1%}", ha = "center", va = "center", fontsize = 7.5, color = "black")
    plt.colorbar( im, ax = ax, label = "Monthly Return")
    ax.set_title( f"Monthly Return Heatmap : [{label}]", fontsize = 13)
    
    plt.tight_layout()
    plt.show()
        