#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 18:13:08 2026

@author: jaykhandelwal
"""

import yfinance as yf
import pandas as pd

def load_data(ticker="SPY", start="2000-01-01", end=None):
    # Returns a clean DataFrame with: Close: adjusted close price (auto_adjust=True) Returns: daily % returns
  
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

# yfinance sometimes returns a multi-level column if we download multiple tickers
# using auto_adjust=True already gives adjusted close so no need to adjust manually

    close = raw["Close"]
    
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]    # just take first column if it's a dataframe for some reason

    df = pd.DataFrame({"Close": close})
    
# pct_change gives us daily % return = (today - yesterday) / yesterday

    df["Returns"] = df["Close"].pct_change() 
    
    df.dropna(inplace=True)    # first row will always be NaN because there's no previous day
   
    print(f"Loaded {len(df)} rows for {ticker} from {df.index[0].date()} to {df.index[-1].date()}")
   
    return df
    

#.dropna() - removes rows with missing values
# inplace=True - modifying the original dataframe directly (updates automatically)
# progress=Fale - removes the download progress spam
    # .iloc[] - in pandas is used to select all rows (:) and the first column (index 0) from a DataFrame, returning the result as a Series. This integer-based indexing operator acts as a slice, and it is commonly used to extract specific data columns based on their numerical position rather than their label.