#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:50:11 2026

@author: jaykhandelwal
"""

# Multi-asset portfolio engine
# Applies the same MA Crossover strategy to multpile tickers independently
# Then combines them into one portfolio using equal weighting
# This shows diversification - not all assests crash at the same time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf 

from strategy import moving_average_strategy
from backtester import backtest
from performance import summarize_performance, compute_drawdown

def load_multi_asset(tickers, start = "2000-01-01", end = None):
    
    print(f"\n [PORTFOLIO] Loading Data for : {tickers}.....")
    asset_data = {}
    
    for ticker in tickers:
        try:
            raw = yf.download(ticker, start = start, end = end, auto_adjust = True, progress = False)
            close = raw["Close"]
            
            if isinstance (close, pd.DataFrame):
                close = close.iloc[:,0]
                
            df = pd.DataFrame({"Close" : close})
            df ["Returns"] = df["Close"].pct_change()
            df.dropna(inplace = True)
            
            asset_data[ticker] = df
            print(f" {ticker}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()}")
            
        except Exception as e:
            print(f" {ticker} : Failed to Load - {e}")
            
    return asset_data

def run_portfolio(asset_data, short_w = 50, long_w = 200, transaction_cost = 0.001, weights = None):
    """
    Applies MA Crossover strategy to each asset independently
    Then combines daily net returns using the given weights ( equal by default)
    
    This works because not all assets move together. When equities crash, bonds often rise and gold holds the vaue
    So the combined portfolio smooths out the worst drawdowns.
    """
    
    tickers = list(asset_data.keys())
    
    if weights is None: #Equal weights (every asset gets the same allocation)
        w = 1.0 / len(tickers)
        weights = {t: w for t in tickers}
        
    print(f"\n[PORTFOLIO] Running strategy on each assets.....")
    print(f" Weights: {weights}")
    
    strategy_returns = {}
    market_returns = {}
    
    for ticker, df in asset_data.items():
        sig = moving_average_strategy(df, short_window = short_w, long_window = long_w)
        bt = backtest(sig, transaction_cost = transaction_cost)
        
        strategy_returns[ticker] = bt["Net_Strategy_return"]
        market_returns[ticker] = bt["Returns"]
        
        
    # Aligning all returns series to a common date index
    strat_df = pd.DataFrame(strategy_returns).dropna()
    market_df = pd.DataFrame(market_returns).dropna()
        
    # Weighted sum of daily returns
    w_series = pd.Series(weights)
    w_series = w_series / w_series.sum()    # normalizing the weight sum to 1
        
    portfolio_returns = strat_df.dot(w_series)    # dot product = weighted sum
    benchmark_return = market_df.dot(w_series)     # same weight applies to B&H 
        
        
    # Building a combined DataFrame
    combined = pd.DataFrame({"Net_Strategy_return": portfolio_returns, "Returns": benchmark_return})
        
    combined["Cumulative_Strategy"] = ( 1 + combined["Net_Strategy_return"]).cumprod()
    combined["Cumulative_Market"] = ( 1 + combined["Returns"]).cumprod()
    combined["Signal"] = 1    # place - holder so that period_report doesn't break


    return combined, strat_df, market_df

def plot_portfolio_vs_spy(portfolio_df, spy_df, label = "Portfolio vs SPY Alone"):
    """
    Comparing the multi-aset portfolio equity curve against SPY strategy alone
    that shows the diversification benefits
    """
    
    from performance import compute_drawdown
    
    spy_strat_cum = (1 + spy_df["Net_Strategy_return"]).cumprod()
    port_cum = (1 + portfolio_df["Net_Strategy_return"]).cumprod()
    market_cum = (1 + portfolio_df["Returns"]).cumprod()
    
    # Index Align 
    common_idx = port_cum.index.intersection(spy_strat_cum.index)
    
    fig, axes = plt.subplots(2,1, figsize = (14,8), sharex = True)
    
    # Equity Curve
    axes[0].plot(common_idx, market_cum.loc[common_idx].values,label = "Equal Weighted Benchmark (B&H)", color = "lightcoral", linewidth = 1.2)
    axes[0].plot(common_idx, spy_strat_cum.loc[common_idx].values, label = "SPY Strategy Alone", color = "steelblue", linewidth = 1.5, linestyle = "--") 
    axes[0].plot(common_idx, port_cum.loc[common_idx].values, label = " Multi Assest Portfolio Strategy", color = "darkorange", linewidth = 1.8)
    axes[0].set_title(f"Equity Curve - {label}", fontsize = 13)
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(fontsize = 9)
    axes[0].grid(True, alpha = 0.25)
    
    # Drawdowns
    dd_mkt = compute_drawdown(market_cum.loc[common_idx])
    dd_spy = compute_drawdown(spy_strat_cum.loc[common_idx])
    dd_port = compute_drawdown(port_cum.loc[common_idx])
    
    axes[1].fill_between(common_idx, dd_mkt.values, 0, alpha = 0.2, color = "lightcoral", label = "Benchmark DD")
    axes[1].plot(common_idx, dd_spy.values, label = " SPY Strategy DD", color = "steelblue", linewidth = 1.4, linestyle = "--")
    axes[1].plot(common_idx, dd_port.values, label = "Portfolio Strategy DD", color = "darkorange", linewidth = 1.6)
    
    axes[1].set_title("Drawdowns", fontsize = 13)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[1].legend(fontsize = 9)
    axes[1].grid(True, alpha = 0.25)
    
    plt.tight_layout()
    plt.show()
    
def print_portfolio_metrics(portfolio_df, spy_bt, label = "Portfolio"):
    """
    Side-byside metrics : Multi-Asset Portfolio vs SPY Strategy vs Market
    """
    port_m = summarize_performance(portfolio_df, "Net_Strategy_return", "Cumulative_Strategy")
    mkt_m =summarize_performance(portfolio_df, "Returns", "Cumulative_Market")
    
    # SPY-only metrics
    spy_cum = spy_bt.copy()
    spy_cum["Cumulative_Strategy"] = ( 1 + spy_cum["Net_Strategy_return"]).cumprod()
    spy_cum["Cumulative_Market"] = ( 1 + spy_cum["Returns"]).cumprod()
    spy_m = summarize_performance(spy_cum, "Net_Strategy_return", "Cumulative_Strategy")
    
    keys = ["CAGR", "Annual_Volatility", "Sharpe", "Sortino", "Max_Drawdown", "Win_Rate_Daily"]
    
    print(f"\n ======= Portfolio Metrics Comparison - {label} ====== ")
    print(f"\n{'Metric':<22}  {'Portfolio':>12}  {'SPY Only':>12}  {'Benchmark':>12}")
    print("-" * 50)
    
    for k in keys:
        p_val = str(port_m.get(k, "N/A"))
        s_val = str(spy_m.get(k, "N/A"))
        m_val = str(mkt_m.get(k, "N/A"))
        print(f"{k:<22} {p_val:>12} {s_val:>12} {m_val:>12}")
    
    
    
    
    