# Quant Momentum Screener

A research-first quantitative stock screener. It ranks a supplied stock universe using multi-period momentum, trend, volatility and risk-adjusted measures.

## Safety
This is a research/backtesting tool, not an automatic trading system. Validate data, costs, slippage and out-of-sample performance before using real capital.

## Run
Install dependencies:
`pip install -r requirements.txt`

Run:
`python main.py`

## Aggressive Plan 2 comparison

`aggressive_plan2_compare.py` compares the current Plan 2 implementation (Top 100 by 12-2 momentum, then lowest-ID Top 50) with an aggressive concentrated variant (5 stocks). The aggressive variant requires price > 50-day SMA > 200-day SMA and ranks positive 3M/6M/12M momentum (30/30/40 weights), penalized by the latest 1M return and divided by annualized volatility. Both strategies use identical historical membership, completed month-end prices and 0.50% turnover cost. The GitHub Actions run exports summary, monthly returns and selection audits as CSV artifacts.

Research only. The 5-stock portfolio is more concentrated and can suffer materially larger drawdowns. Review unseen-test results, costs, liquidity and paper-trading performance before considering live capital. The comparison uses fractional portfolio weights for return calculations; it is not an exact whole-share ₹20,000 execution simulation.
