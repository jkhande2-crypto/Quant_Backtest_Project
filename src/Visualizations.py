#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:06:51 2026

@author: jaykhandelwal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_full_equity_curve (bt_df, train_end, label ="Strategy vs Market"):
    
    cum_strat = ( 1 + bt_df["Net_Strategy_return"]).cumprod()
    cum_market = ( 1 + bt_df["Returns"]).cumprod()
    
    fig, ax = plt.subplots(figsize = (14,4))
    
    ax.plot(cum_market.index, cum_market.values, label = " Market ( Buy & Hold)", color = "lightcoral", linewidth = 1.2, alpha = 0.8) 
    ax.plot(cum_strat.index, cum_strat.values, label = "MA Crossover Strategy", color = "steelblue", linewidth =1.5)
    
    # V-lines at IS/OOS boundary
    split_date = pd.Timestamp(train_end)
    ax.axvline( split_date, color ="black", linestyle = "--", linewidth = 1.2, label = f" IS/OOS Split ({train_end})")
    
    # Shade the two regions lightly
    ax.axvspan(cum_strat.index[0], split_date, alpha = 0.04, color = "steelblue", label = "In-Sample")
    ax.axvspan(split_date, cum_strat.index[-1], alpha = 0.04, color = "darkorange", label = "Out-Of-Sample")
    
    ax.set_title(f"Full Period Equity Curve : {label}", fontsize = 13)
    ax.set_ylabel( "Growth of $1")
    ax.legend(fontsize = 9 )
    ax.grid( True, alpha = 0.25)
    
    plt.tight_layout()
    plt.show()
    
def plot_drawdown_curve( bt_df, train_end, label = "Drawdown Comparison"):
    """
    Drawdown chart showing both strategy and market drawdown.
    A good strategy should have a shallower drawdowns than the market.
    """
    
    from performance import compute_drawdown
    
    cum_strat = (1 + bt_df["Net_Strategy_return"]).cumprod()
    cum_market = (1+ bt_df["Returns"]).cumprod()
    
    dd_strat = compute_drawdown(cum_strat)
    dd_market = compute_drawdown (cum_market)
    
    fig, ax = plt.subplots(figsize = (14,4))
    
    ax.fill_between(dd_market.index, dd_market.values, 0, alpha=0.3, color="lightcoral", label="Market Drawdown")
    ax.fill_between(dd_strat.index,  dd_strat.values,  0, alpha=0.3, color="steelblue", label="Strategy Drawdown")
    
    ax.axvline(pd.Timestamp(train_end), color = "black", linestyle = "--", linewidth = 1.2, label = f"IS/OOS Split ({train_end})" )
    
    ax.set_title(f"Drawdown: {label}", fontsize = 13)
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter( plt.FuncFormatter( lambda y, _ :f"{y:.0%}"))
    ax.legend(fontsize = 9)
    ax.grid(True, alpha = 0.25)
    
    plt.tight_layout()
    plt.show()
    
def plot_rolling_sharpe( bt_df, window = 252, train_end = None, label ="Rolling Sharpe" ):
    """
     Rolling Sharpe ratio using trailing window (default = 252 trading days = 1 year )
     Shows whether the strategy's edge is consistent over time or  concentrated in a few periods.
     A flat, positive rolling sharpe = consistent edge.
     A spiky or decling sharpe = the edge is not stable.
    """
    
    r = bt_df["Net_Strategy_return"].dropna()
    
    rolling_mean = r.rolling(window).mean() * 252
    rolling_std = r.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_std
    
    fig, ax =plt.subplots(figsize = (14,4), sharex = True)
    
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color = "steelblue", linewidth = 1.3, label = f"Rolling Sharpe ({window}d)")
    ax.axhline( 0, color = "black", linewidth = 0.8, linestyle = "--")
    ax.axhline( rolling_sharpe.mean(), color = "darkorange", linewidth = 1.0, linestyle = "--", label = f"Mean Sharpe = {rolling_sharpe.mean():.4f}") 
    
    if train_end:
        ax.axvline(pd.Timestamp(train_end), color = "black", linestyle = "--", linewidth = 1.2, label = f" IS/OOS Split ({train_end})")
    
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0, where = (rolling_sharpe > 0), alpha = 0.15, color = "green")
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0, where = (rolling_sharpe < 0), alpha = 0.15, color ="red")
    
    ax.set_title( f"Rolling {window} - Day Sharpe Ratio - {label}", fontsize = 13)
    ax.set_ylabel("Sharpe Ratio")
    ax.legend( fontsize = 9)
    ax.grid(True, alpha = 0.25)
    
    plt.tight_layout()
    plt.show()
    
def plot_signal_over_time (bt_df, label = "Signal (Long/Flat)"):
    """
    Shows when the strategy was invested over the full period ( SIgnal = 1)
    Blue bars = invested in the market
    White Gaps = sitting in cash
    To visually verify the strategy got out during the major crashes.
    
    """
    
    fig, axes = plt.subplots(2,1, figsize = (14,6), sharex = True )
    
    # Top : Price with signal shading
    close = bt_df["Close"]
    signal = bt_df["Signal"]
    
    axes[0].plot(close.index, close.values, color = "black", linewidth = 0.9, label ="SPY Price")
    axes[0].fill_between( close.index, close.min(), close.max(), where = ( signal == 1), alpha = 0.2, color = "steelblue", label = "Long ")
    axes[0].set_title(f" Price with Signal Overlay - {label}", fontsize = 13)
    axes[0].legend( fontsize = 9)
    axes[0].grid( True, alpha = 0.2)
    
    # Bottom : Signal as 0/1 bar
    axes[1].fill_between(signal.index, 0, signal.values,color = "steelblue", alpha = 0.6, step = "pre")
    axes[1].set_title( "Raw Signal ( 1 = Long, 0 = Flat)", fontsize = 11)
    axes[1].set_ylabel("Signal")
    axes[1].set_ylim( -0.1, 1.3)
    axes[1].grid (True, alpha = 0.2)
    
    plt.tight_layout()
    plt.show()

def plot_oos_comparison (test_base, test_best, base_label, best_label):
    """
    Side-by-side OOS equity curve and drawdowns for baseline vs best params
    To prove the parameter search added value
    
    """
    
    from performance import compute_drawdown
    
    fig, axes = plt.subplots(2,1, figsize = (14,4), sharex = True)
    
    # Equity Curve
    axes[0].plot(test_base["Cumulative_Market"].values, label =" Market (B&H)", color = "black", linewidth = 1.1)
    axes[0].plot(test_base["Cumulative_Strategy"].values, label = base_label, color = "steelblue", linewidth = 1.6, linestyle = "--")
    axes[0].plot(test_base["Cumulative_Strategy"].values, label = best_label, color = "darkorange", linewidth = 1.6)
    axes[0].set_title("Out-Of-Sample Equity Curve", fontsize = 13)
    axes[0].set_ylabel( "Growth of $1")
    axes[0].legend( fontsize = 9)
    axes[0].grid (True, alpha = 0.25)
    
    # Drawdown
    dd_mkt = compute_drawdown(test_base["Cumulative_Market"])
    dd_baseline = compute_drawdown(test_base["Cumulative_Strategy"])
    dd_best = compute_drawdown(test_best["Cumulative_Strategy"])
    
    axes[1].plot(dd_mkt.values, label = "Market DD", color = "black", linewidth = 1.1)
    axes[1].plot(dd_baseline.values, label = base_label, color = "steelblue", linewidth = 1.6)
    axes[1].plot(dd_best.values, label = best_label, color = "darkorange", linewidth = 1.6)
    axes[1].set_title( "Out-Of-Sample", fontsize = 13)
    axes[1].set_ylabel("Drawdwon (%)")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter( lambda y, _ : f"{y:.0%}"))
    axes[1].legend(fontsize = 9)
    axes[1].grid (True, alpha = 0.25)
    
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    