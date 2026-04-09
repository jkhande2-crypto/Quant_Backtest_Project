#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 18:11:41 2026

@author: jaykhandelwal
"""

# Complete pipeline: Data -> Strategy -> Backtest -> IS/OOS -> Sensitivity Analysis -> Exposure -> Visualiztions -> Portfolio Engine 

import sys
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


sys.path.append(os.path.abspath("src"))

from data_loader import load_data
from strategy import moving_average_strategy 
from backtester import backtest 
from performance import summarize_performance, compute_drawdown, print_metrics, period_report
from Sensitivity import run_sensitivity_grid, plot_sensitivity_heatmaps, plot_3d_sensitivity
from Exposure import compute_exposure_stats, plot_annual_returns, plot_monthly_heatmap
from Visualizations import (plot_full_equity_curve, plot_drawdown_curve, plot_rolling_sharpe, plot_signal_over_time, plot_oos_comparison)
from Portfolio import (load_multi_asset, run_portfolio, plot_portfolio_vs_spy, print_portfolio_metrics)


# -----------------

# Settings 
# -----------------

TICKER = "SPY"
START = "2000-01-01"
TRAIN_END = "2015-12-31" 
TEST_START = "2016-01-01"

TRANSACTION_COST = 0.001  # 10 bps per trade

PORTFOLIO_TICKERS = ["SPY", "QQQ", "GLD", "IEF"]    


# --------------------------

# Helpers

# --------------------------


def run_pipeline( raw_data, short_w, long_w, tc = TRANSACTION_COST) :
    df = moving_average_strategy (raw_data, short_window = short_w, long_window = long_w)
    df = backtest(df, transaction_cost = tc)
    return df 


def rescale_cumulative (sub):
    # Rescale cumulative curves so the period starts at 1. 
    # This makes CAGR/drawdown correct for that sub-period.
    
    out = sub.copy()
    for col in ["Cumulative_Market" , "Cumulative_Strategy"] :
        out[col] = out[col] / out[col].iloc[0]
    return out 
    
def print_metrics (title, strat, mkt):
        def pct (x) :
            return f"{x * 100: .2f}%"
        print (f"\n ====== {title} ======")
        print (f" Strategy | CAGR {pct(strat["CAGR"])} | Sharpe {strat["Sharpe"]: .2f} | MaxDD {pct(strat ["Max_Drawdown"])}")
        print (f" Market | CAGR {pct(mkt["CAGR"])} | Sharpe {mkt["Sharpe"]: .2f} | MaxDD {pct(mkt ["Max_Drawdown"])}")
        
def evaluate_in_sample_oos (bt_full, train_end, test_start):
    # Compute metrics for in-sample and out-of-sample slices.
    
 
    train = rescale_cumulative( bt_full.loc[: train_end].copy())
    test = rescale_cumulative( bt_full.loc[ test_start :].copy())
    
    train_strat = summarize_performance (train, "Net_Strategy_return", "Cumulative_Strategy")
    train_mkt = summarize_performance ( train, "Returns", "Cumulative_Market")
    
    test_strat = summarize_performance (test, "Net_Strategy_return", "Cumulative_Strategy")
    test_mkt = summarize_performance ( test, "Returns", "Cumulative_Market")
    
    return train, test, train_strat, train_mkt, test_strat, test_mkt



# -------------------------

# 1 - LOAD DATA

# --------------------------

print("\n" + "=" * 60)
print (" LOAD DATA")
print("=" * 60)

data = load_data(ticker = TICKER, start = START)

# -------------------------------------------

# 2 - BASELINE: 50/200 IS AND OOS

# -------------------------------------------

print("\n" + "="*60)
print("BASELINE BACKTEST (MA 50/200)")
print("="*60)



BASE_SHORT, BASE_LONG = 50, 200
bt_base = run_pipeline ( data, BASE_SHORT, BASE_LONG )

train_base, test_base, train_strat_b, train_mkt_b, test_strat_b, test_mkt_b = evaluate_in_sample_oos ( bt_base, TRAIN_END, TEST_START)

print_metrics(f" IN_Sample | Baseline MA { BASE_SHORT} / { BASE_LONG})", train_strat_b, train_mkt_b)

print_metrics(f" Out_of_Sample  | Baseline MA { BASE_SHORT} / { BASE_LONG})", test_strat_b, test_mkt_b)

# ---------------------------------------------------------------------

# 3 — CRISIS PERIOD ANALYSIS (on full backtest, default 50/200)

# ---------------------------------------------------------------------


print("\n========== CRISIS PERIOD ANALYSIS (Full Sample, 50/200) ==========")
 
period_report(bt_base, "2001-01-01", "2002-12-31", label="Dot-Com Crash")
period_report(bt_base, "2008-01-01", "2009-06-30", label="GFC 2008")
period_report(bt_base, "2020-01-01", "2020-12-31", label="COVID 2020")
period_report(bt_base, "2022-01-01", "2022-12-31", label="Rate Hike 2022")
 
# -----------------------------------------------

# 4 - IN-SAMPLE PARAMETER SEARCH

# -----------------------------------------------

# training data is used here 


print("\n========== IN-SAMPLE PARAMETER SEARCH ==========")

short_list = [20, 50, 100]
long_list = [100, 150, 200, 300]

results = []

for s in short_list:
    for l in long_list:
         if s >=l:
             continue     # short MA must always be shorter than long MA
         
         bt = run_pipeline( data, s, l )
         
         # Only look at IS slice
         
         train_slice = rescale_cumulative( bt.loc[:TRAIN_END].copy())
         
         m = summarize_performance (train_slice, "Net_Strategy_return", "Cumulative_Strategy")
         
         results.append({
             "short": s,
             "long": l,
             "train_sharpe": m['Sharpe'],
             "train_cagr": m["CAGR"],
             "train_sortino": m["Sortino"],
             "train_max_dd": m["Max_Drawdown"]
             })
         
results_df = pd.DataFrame(results).sort_values("train_sharpe", ascending = False).reset_index(drop=True)
         
print ( "\n Top Parameter Sets by IN-SAMPLE Sharpe:")
print ( results_df.to_string(index = False))
      
# Pick best
    
best_short = int(results_df.iloc[0]["short"])
best_long = int( results_df.iloc[0]["long"])
         
print( f"\n Chosen Best IN-SAMPLE parameters : Short = {best_short} | Long = {best_long}")

         
bt_best = run_pipeline( data, best_short, best_long)

train_best, test_best, train_strat_best, train_mkt_best, test_strat_best, test_mkt_best = evaluate_in_sample_oos (bt_best, TRAIN_END, TEST_START)
         
print_metrics(f"In-Sample | Best MA { best_short} / {best_long})", train_strat_best, train_mkt_best)

print_metrics(f"Out-of-Sample | Best MA { best_short} / {best_long})", test_strat_best, test_mkt_best)


# ------------------------------------

# 5 — IS vs OOS COMPARISON TABLE

# ------------------------------------

 
print("\n========== IS vs OOS COMPARISON — BEST PARAMS ==========")

print("-" * 46)
 
keys = ["CAGR", "Annual_Return", "Annual_Volatility", "Sharpe", "Sortino", "Max_Drawdown", "Win_Rate_Daily"]

print(f"\n{'Metric':<22}  {'IS':>10}  {'OOS':>10}")

for k in keys:
    print(f"{k:<22}  {str(train_strat_best.get(k, 'N/A')):>10}  {str(test_strat_best.get(k, 'N/A')):>10}")
 

# ---------------------------------------

# 6 - PARAMETER SENSITIVITY ANALYSIS

# ---------------------------------------

# Wide grid - only on IS data, never touches OOS
is_data = data.loc[:TRAIN_END].copy()

sensitivity_df = run_sensitivity_grid( is_data, short_range = range(10,130,10), long_range = range(100,425,25), transaction_cost = TRANSACTION_COST)

plot_sensitivity_heatmaps(sensitivity_df)
plot_3d_sensitivity(sensitivity_df, metric = "Sharpe")
plot_3d_sensitivity(sensitivity_df, metric = "CAGR")
plot_3d_sensitivity(sensitivity_df, metric = "Max_DD")


# ---------------------------------------

# 7 - EXPOSURE ANALYSIS

# ---------------------------------------

compute_exposure_stats(bt_base, label = f"Full Period - MA ({BASE_SHORT} / {BASE_LONG})")
compute_exposure_stats(bt_base.loc[: TRAIN_END], label = f"In-Sample - MA ({ BASE_SHORT} / {BASE_LONG})")
compute_exposure_stats(bt_base.loc[TEST_START :], label = f"Out-Of-Sample - MA ({BASE_SHORT} / {BASE_LONG})")

plot_annual_returns(bt_base, label = f" MA ({ BASE_SHORT} / {BASE_LONG}) - Full Period")
plot_monthly_heatmap(bt_base, col = "Net_Strategy_return", label = f" MA ({BASE_SHORT} / {BASE_LONG}) - Strategy")

# ------------------------------------------

# 8 - FULL VISAULIZATION

# ------------------------------------------

# 1. Full Equity Curve with IS / OOS Split Marked
plot_full_equity_curve(bt_base, TRAIN_END, label = f" MA ({BASE_SHORT} / {BASE_LONG})")

# 2. Drawdown Comparison
plot_drawdown_curve(bt_base, TRAIN_END, label = f"MA ({BASE_SHORT} / {BASE_LONG})")

# 3. Rolling Sharpe
plot_rolling_sharpe(bt_base, window = 252, train_end = TRAIN_END, label = f" MA ({BASE_SHORT} / {BASE_LONG})")

# 4. Signal Overlay on Price Chart
plot_signal_over_time(bt_base, label = f"MA ({BASE_SHORT} /{BASE_LONG})")

# 5. OOS COmparison: Baseline v/s Best Params
test_base_plot = rescale_cumulative(bt_base.loc[TEST_START:].copy())
test_best_plot = rescale_cumulative(bt_base.loc[TEST_START:].copy())

plot_oos_comparison( test_base_plot, test_best_plot, base_label = f" Baseline MA ({BASE_SHORT} / {BASE_LONG})", best_label = f" Best MA ({BASE_SHORT} / {BASE_LONG})")


# ------------------------------

# 9 - PORTFOLIO ENGINE

# ------------------------------

asset_data = load_multi_asset(PORTFOLIO_TICKERS, start = START)

portfolio_df, strat_returns_df, market_returns_df = run_portfolio(asset_data, short_w = BASE_SHORT, long_w = BASE_LONG, transaction_cost = TRANSACTION_COST)

spy_bt = bt_base.copy()

print_portfolio_metrics(portfolio_df, spy_bt, label = "Equal Weighted 4 - Assets v/s SPY Alone ")

plot_portfolio_vs_spy(portfolio_df, spy_bt, label = " Multi-Asset Portfolio v/s SPY Alone ")

print(""" 
      What this project covers:
          1. data loading and cleaning
          2. MA crossover strategy with look--ahead basis control
          3. Transaction cost modeling
          4. In-sampel / Out-of-sample split and evaluation
          5. Crisis period analysis ( DOt-com, GFC, COvid, 2022)
          6. Parameter grid search (IS only)
          7. Parameter sensitivity heatmaps ( Sharpe, CAGR, Max DD)
          8. Exposure and trade analysis
          9. Annual Returns bar chart + montly returns heatmap
          10. Full equity curve with IS / OOS boundary marked
          11. Drawdown comparison chart
          12. Rolling Sharpe Ratio ( 252-day window)
          13. Signal overlay on price chart
          14. OOS equity curve comparison (baseline v/s best parameters)
          15. Multi-Asset portfolio engine ( SPY + QQQ + GLD + IEF)
          """)

