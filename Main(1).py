#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 17:10:25 2026

@author: jaykhandelwal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 18:11:41 2026

@author: jaykhandelwal
"""

import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("src"))

from data_loader import load_data
from strategy import moving_average_strategy 
from backtester import backtest 
from performance import summarize_performance, compute_drawdown, period_report

# 1. Load Long History ( includes 2008)
data = load_data(ticker = "SPY", start = "2000-01-01")  #function execution

# 2. Strategy 
data = moving_average_strategy(data, short_window = 50, long_window = 200)

# 3. Backtest
data = backtest(data,transaction_cost = 0.001)

# 4. Full - Period Metrics
strategy_metrics = summarize_performance(data, "Net_Strategy_Returns", "Cumulative_Strategy")
market_metrics = summarize_performance(data, "Returns", "Cumulative_Market")

print("\n ====== FULL PERIOD METRICS ======")
print("Strategy:", strategy_metrics)
print("\n Market:", market_metrics)

# 5. Drawdown Series ( for plotting)
data[ "DD_Strategy"] = compute_drawdown(data["Cumulative_Strategy"])
data[ "DD_Market"] = compute_drawdown(data["Cumulative_Market"])

# 6. Plotting Equity Curves

plt.figure(figsize=(10,6))
plt.plot(data['Cumulative_Market'],label='Market ( Buy & Hold )')
plt.plot(data['Cumulative_Strategy'],label='MA Strategy ( Net )')
plt.title("Equity Curve: Strategy vs Market")
plt.legend()
plt.show()

# 7. Plotting Drawdowns
plt.figure(figsize=(10,6))
plt.plot(data['DD_Market'],label='Market Drawdown')
plt.plot(data['DD_Strategy'],label='Strategy Drawdown')
plt.title("Drawdown: Strategy vs Market")
plt.legend()
plt.show()

# 8. Crash Report
period_report(data, "2008-01-01", "2009-06-30", label = "GFC 2008 - 2009")
period_report(data, "2020-02-01", "2020-04-30", label = "COVID Crash ( Feb - Apr 2020 )")

#Show Result
#print(data.head()) shows the first 5 rows of our data frame. it's like "showing the beginning of my dataset." Quickly check if data is loaded properly, verifying columns and can see the formats.

#print(data.tail()) similary, shows the last 5 rows of our dataframe. it's like "showing the most recent data."

#print(data[['Close','MA_Short','MA_Long','Signal']].tail())

#print(data[['Cumulative_Market','Cumulative_Strategy',]].tail())



