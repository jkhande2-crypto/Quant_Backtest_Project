# Quant Backtest Project
### Moving Average Crossover Strategy | SPY | 2000 - 2026

A fully modular backtesting pipeline built in Python from scratch. No pre-built backtesting libraries. The project tests whether the classic 50/200 Moving Average crossover strategy holds up on SPY across 26 years of data, with rigorous in-sample / out-of-sample evaluation, parameter sensitivity analysis, crisis period stress testing, and a multi-asset portfolio engine.

---

## The Question

Does the 50/200 MA crossover strategy actually work on SPY, or is it just a textbook example that falls apart in the real world?

---

## Strategy Logic

A long signal is generated when the short-term MA crosses above the long-term MA. The strategy moves to cash when it crosses below. A one-day signal shift is applied on every position to prevent look-ahead bias. Transaction costs are charged on every entry and exit.

| Parameter | Value |
|---|---|
| Baseline Short MA | 50 days |
| Baseline Long MA | 200 days |
| Transaction Cost | 0.10% per trade |
| Data Source | SPY via yfinance |
| Full Period | 2000-01-04 to 2026-04-08 |
| Total Observations | 6,604 daily rows |

---

## Key Results

### In-Sample vs. Out-of-Sample (MA 50/200)

| Metric | In-Sample | Out-of-Sample |
|---|---|---|
| CAGR | 7.51% | 8.89% |
| Sharpe Ratio | 0.7109 | 0.6311 |
| Sortino Ratio | 0.7421 | 0.6584 |
| Max Drawdown | -17.31% | -33.72% |
| Annual Volatility | 11.05% | 15.10% |
| Win Rate (Daily) | 35.16% | 45.54% |

### vs. Market (Buy and Hold)

| Metric | In-Sample | Out-of-Sample |
|---|---|---|
| Market CAGR | 4.31% | 14.44% |
| Market Sharpe | 0.30 | 0.83 |
| Market Max DD | -55.19% | -33.72% |

The strategy delivers more than double the risk-adjusted return of buy and hold during the in-sample period. OOS Sharpe decay from 0.71 to 0.63 is within acceptable range and does not indicate overfitting.

---

## Crisis Period Performance

| Period | Market Return | Strategy Return | Strategy Max DD | Avg Exposure |
|---|---|---|---|---|
| Dot-Com Crash (2001-2002) | -29.53% | -1.95% | -6.99% | 4.00% |
| GFC 2008 (2008-2009) | -34.19% | +0.17% | -3.00% | 2.39% |
| COVID 2020 | +17.24% | -5.30% | -33.72% | 73.91% |
| Rate Hike 2022 | -18.65% | -8.90% | -12.87% | 19.92% |

The strategy avoided the Dot-Com and GFC crashes almost entirely due to very low market exposure (under 5%). COVID 2020 is the key failure period. 73.91% exposure going into the crash meant the strategy absorbed the full drawdown. That is a real weakness and is acknowledged explicitly rather than ignored.

---

## Trade Exposure Analysis

| Metric | Full Period | In-Sample | Out-of-Sample |
|---|---|---|---|
| Time Invested | 70.35% | 62.95% | 81.90% |
| Time in Cash | 29.65% | 37.05% | 18.10% |
| Number of Trades | 14 | 9 | 6 |
| Avg Holding Period | 331.9 days (15.8 months) | 281.4 days (13.4 months) | 352.2 days (16.8 months) |
| Longest Hold | 910 days (43.3 months) | 910 days (43.3 months) | 667 days (31.8 months) |
| Shortest Hold | 12 days (0.6 months) | 12 days (0.6 months) | 9 days (0.4 months) |

14 trades over 26 years. This is a very low turnover strategy, approximately 0.54 round trips per year.

---

## Parameter Sensitivity

Grid search across 153 valid (Short MA, Long MA) combinations on in-sample data only. OOS data was never touched during this process.

| Short MA | Long MA | Sharpe | CAGR | Max DD |
|---|---|---|---|---|
| 30 | 225 | 0.7408 | 7.76% | -17.31% |
| 50 | 225 | 0.7291 | 7.67% | -17.31% |
| 50 | 200 | 0.7109 | 7.51% | -17.31% |
| 60 | 225 | 0.7067 | 7.39% | -17.31% |
| 30 | 200 | 0.7047 | 7.34% | -17.31% |
| 70 | 400 | 0.6989 | 7.78% | -18.61% |
| 40 | 225 | 0.6972 | 7.29% | -17.85% |

The 50/200 baseline sits inside a broad high-performance plateau in the 30-70 short and 200-275 long region. A strategy that only works at one specific parameter setting is overfitting. A strategy with a broad green region is robust. This one has a broad green region.

---

## Multi-Asset Portfolio Engine

The same MA crossover strategy applied independently to four assets and combined with equal weighting (25% each).

| Metric | Portfolio | SPY Strategy Only | Equal Weight Benchmark |
|---|---|---|---|
| CAGR | 8.43% | 7.99% | 10.85% |
| Sharpe | 0.9870 | 0.6656 | 0.9901 |
| Sortino | 1.2488 | 0.6800 | 1.3118 |
| Max Drawdown | -16.17% | -33.72% | -28.01% |
| Annual Volatility | 8.53% | 12.78% | 10.99% |

Portfolio Sharpe of 0.987 vs. SPY-only Sharpe of 0.666 is a 48.3% improvement in risk-adjusted returns. Max Drawdown cut from -33.72% to -16.17%. Diversification across uncorrelated assets works exactly the way the theory says it should.

---

## Known Weaknesses

This project is built to be honest, not just to look good.

- COVID 2020 remains the biggest failure. High exposure going into a fast crash means the slow-moving MA crossover cannot exit in time.
- The strategy underperforms in strong trending bull markets where being out of the market is costly. The 2013-2019 and 2023-2024 periods are examples where buy and hold wins comfortably.
- The 50/200 MA crossover is one of the most widely known signals in markets. Part of its historical edge may reflect self-fulfilling behavior from institutional adoption rather than genuine alpha.
- Transaction costs are modeled as a flat 0.10% but real-world slippage, especially during low-liquidity crisis periods, is not modeled.
- No formal statistical significance test is applied to the IS Sharpe. A t-test on the daily return series would strengthen the conclusion.

---

## Project Structure

```
Quant_Backtest_Project/
|
|-- main.py                    Master pipeline, runs everything in order
|
|-- src/
    |-- data_loader.py         Downloads and cleans price data via yfinance
    |-- strategy.py            MA crossover signal generation with look-ahead bias control
    |-- backtester.py          Applies signal, models transaction costs, builds equity curve
    |-- performance.py         Sharpe, Sortino, CAGR, Max DD, Win Rate calculations
    |-- Sensitivity.py         Parameter grid search, 2D heatmaps, 3D surface plot
    |-- Exposure.py            Trade exposure stats, annual returns chart, monthly heatmap
    |-- Visualizations.py      Equity curve, drawdown, rolling Sharpe, signal overlay
    |-- Portfolio.py           Multi-asset engine: load, run, combine, compare
```

---

## Visualizations Produced

- Full equity curve with IS/OOS split boundary
- Drawdown comparison chart (strategy vs. market)
- Rolling 252-day Sharpe ratio
- Signal overlay on price chart
- OOS equity curve comparison (baseline vs. best parameters)
- Annual returns bar chart (strategy vs. market, by year)
- Monthly returns heatmap (green/red grid by year and month)
- 2D parameter sensitivity heatmaps (Sharpe, CAGR, Max DD)
- 3D surface plot of Sharpe across parameter space

---

## How to Run

**1. Install dependencies:**

```bash
pip install numpy pandas matplotlib yfinance
```

**2. Clone the repository:**

```bash
git clone https://github.com/jkhande2-crypto/Quant_Backtest_Project.git
cd Quant_Backtest_Project
```

**3. Run the pipeline:**

```bash
python main.py
```

All charts render automatically. Console output includes all performance metrics, crisis analysis, sensitivity results, exposure statistics, and portfolio comparison.

---

## Dependencies

| Library | Purpose |
|---|---|
| numpy | Numerical computation |
| pandas | Data manipulation and time series |
| matplotlib | All chart and visualization rendering |
| yfinance | Market data download |
| itertools | Parameter combination generation |

---

## What This Project Demonstrates

- Look-ahead bias prevention via signal shifting
- In-sample / out-of-sample split methodology
- Transaction cost modeling
- Parameter robustness testing via grid search
- Crisis period stress testing across four major market events
- Sharpe, Sortino, CAGR, Max Drawdown, Win Rate computation
- Trade exposure and holding period analysis
- Diversification benefit quantification across asset classes
- Honest acknowledgment of strategy weaknesses
- Modular, clean Python project structure across 8 dedicated files

---

## Author

Jay Khandelwal
GitHub: https://github.com/jkhande2-crypto
LinkedIn: https://linkedin.com/in/jay-khandelwal30

---

This project is for educational and research purposes only. Nothing in this repository constitutes financial advice or a recommendation to trade any security.
