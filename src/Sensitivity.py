#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 03:39:33 2026

@author: jaykhandelwal
"""

# Runs a wide grid search across many short/long MA combinations
# Then draws heatmaps so we can visually see which parameter regions are stable
# Stable = nearby parameters also gives good results not just one lucky combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from mpl_toolkits.mplot3d import Axes3D


def run_sensitivity_grid(data, short_range, long_range, transaction_cost = 0.001): 
    """
    Tests every valid ( Short, Long) MA pairs on the given data.
    Returns a DataFrame with Sharpe, CAGR, MaxDD for each combo.
    We use this to check if our best parameters are genuinely good or just got lucky in one tiny corner of the parameter space.
    """
    
    from strategy import moving_average_strategy
    from backtester import backtest
    
    results = []
    
    all_combos = list(itertools.product(short_range, long_range))
    valid_combos = [(s,l) for s, l in all_combos if s < l]
    
    print(f"\n [SENSITIVITY] Testing {len(valid_combos)} parameters combinations...")
    
    for short_w, long_w in valid_combos:
        
        try:
            df = moving_average_strategy(data, short_window = short_w, long_window = long_w)
            df = backtest(df, transaction_cost = transaction_cost)
            
            r = df["Net_Strategy_return"].dropna()
            
            if len(r)< 100:
                continue
            
            ann_ret = r.mean() * 252
            ann_vol = r.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
            
            
            cum = (1+r).cumprod()
            peak = cum.cummax()
            max_dd = ((cum - peak) / peak ).min()
            
            n_days = len(r)
            start_val = cum.iloc[0]
            end_val = cum.iloc[-1]
            cagr = (end_val / start_val) ** (252 / n_days) - 1
            
            results.append({
                "Short_MA": short_w,
                "Long_MA" : long_w,
                "Sharpe" : round(sharpe, 4),
                "CAGR" : round(cagr, 4),
                "Max_DD" : round(max_dd, 4)
               })
            
        except Exception as e:
            print(f"Failed for short={short_w}, long={long_w} — Error: {e}")
            continue
    results_df = pd.DataFrame(results).sort_values("Sharpe", ascending = False).reset_index(drop = True)
    
    print(f"\n [Sensitiity] Done. {len(results_df)} combinations computed.")
    print("\n Top 15 Combinations by Sharpe: ")
    print(results_df.head(15).to_string(index = False))
    
    return results_df

def plot_sensitivity_heatmaps(results_df) :
    """
    Draws three heatmaps: Sharpe, CAGR and Max Drawdown.
    Each cell = one (short MA, long MA) combination.
    Green = good , Red = bad.
    The goal is to find a GREEN REGION not just a green dot.
    A green regions means the strategy is robust - nearby parameters also works.
    """
    
    metrics = {"Sharpe" : ("RdYlGn", True), # higher is better
               "CAGR" : ("RdYlGn", True),   # higher is better
               "Max_DD" : ("RdYlGn_r", False) # less negative is better, reversed colormap
               }
    
    fig, axes = plt.subplots (1, 3, figsize = (18, 6))
    fig.suptitle(" Parameter Sensitivity HeatMaps ( In - Sample)", fontsize = 14, fontweight = "bold")
    
    for ax, (metric, (cmap, _)) in zip(axes, metrics.items()):
        
        # Pivot to 2D grid: rows = Short MA, columns = Long MA
        
        pivot = results_df.pivot(index = "Short_MA", columns = "Long_MA", values = metric)
        
        im = ax.imshow(pivot.values, cmap=cmap, aspect = "auto")
        
        # Label axes with actual MA Values
        
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation = 45, fontsize = 8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize = 8)
        
        ax.set_xlabel("Long MA")
        ax.set_ylabel("Short MA")
        ax.set_title(metric)
        
        # Put the actual number inside each cell
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[ i, j]
                if not np.isnan(val):
                    ax.text (j, i, f"{val: .2f}", ha = "center", va = "center", fontsize = 7, color = "black" )
                    
        plt.colorbar(im, ax = ax)
        
    plt.tight_layout()
    plt.show()

def plot_3d_sensitivity(results_df, metric  ="Sharpe"):
    """
    3D surface plot of a custom metrics [Sharpe, CAGR, Max DD] across all valid parameter combinations
    X = Short MA, Y = Long MA and Z = chosen Metric. A peak in the surface indicates a robust parameter region.
    
    """
    
    pivot = results_df.pivot( index= "Short_MA", columns = "Long_MA", values = metric)
    
    # build cordination grids matching the pivot dimensions
    x_vals = pivot.index.values
    y_vals = pivot.columns.values
    X,Y = np.meshgrid(y_vals, x_vals)
    Z = pivot.values
    
    fig = plt.figure(figsize = (14,7))
    ax = fig.add_subplot(111, projection = "3d")
    
    surf = ax.plot_surface (X,Y,Z, cmap = "RdYlGn", edgecolor = "none", alpha = 0.90 )
    
    ax.set_title(f"3D Parameter Sensitivity - {metric} (In-Sample)", fontsize =12)
    ax.set_ylabel("Short_MA", fontsize= 12, labelpad= 10)
    ax.set_xlabel("Long_MA", fontsize = 12, labelpad = 10)
    ax.set_zlabel(metric, fontsize = 12, labelpad = 10)
    
    fig.colorbar(surf, ax = ax, shrink = 0.5, aspect = 10, label = metric)
    
    plt.tight_layout()
    plt.show()
    
    
    