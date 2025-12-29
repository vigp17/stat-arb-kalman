# Statistical Arbitrage: Kalman Filter Pairs Trading

## 1. Project Overview
This project implements a **Statistical Arbitrage (Pairs Trading)** strategy that utilizes a **Kalman Filter** to estimate dynamic hedge ratios between cointegrated assets. 

Unlike traditional Ordinary Least Squares (OLS) regression, which assumes a static relationship between asset pairs, the Kalman Filter treats the hedge ratio ($\beta$) as a hidden state variable that evolves over time. This allows the model to adapt to changing market regimes and structural breaks in correlation.

## 2. Methodology
* **Universe Selection:** Tested pairs across Tech, Energy, and Banking sectors. Selected **Gold Miners (GDX)** vs. **Gold (GLD)** due to strong fundamental cointegration.
* **Algorithm:** Implemented a **Kalman Filter** from scratch in Python (NumPy) to estimate the linear state space model:
    $$y_t = \alpha_t + \beta_t x_t + \epsilon_t$$
* **Signal Generation:** Trades are executed when the prediction error (Spread) deviates by **1.25 standard deviations** (Z-Score) from the dynamic mean.
* **Risk Management:** Implemented a volatility-based Stop Loss at 2.5 standard deviations to prevent losses during divergence events.

## 3. Performance Results (Backtest 2020-2024)
After optimizing entry thresholds via sensitivity analysis, the strategy delivered the following results on the GDX/GLD pair:

| Metric | Result |
| :--- | :--- |
| **Total Return** | **+69% (1.69x)** |
| **Annualized Sharpe Ratio** | **0.70** |
| **Strategy Logic** | Mean Reversion |

## 4. Key Files
* `src/kalman.py`: Custom implementation of the Kalman Filter algorithm.
* `src/backtester.py`: Vectorized backtesting engine with spread and transaction logic.
* `optimize.py`: Script used for parameter sensitivity analysis (finding the optimal 1.25 threshold).

## 5. Technology Stack
* **Python:** NumPy, Pandas
* **Data:** Yahoo Finance (yfinance)
* **Visualization:** Matplotlib

---
*Disclaimer: This project is for educational and research purposes only.*