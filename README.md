Delta-Hedged Short Call Backtester

Historical + GBM Monte Carlo Framework

&nbsp;Overview

This project implements a delta-hedged short ATM call strategy using:

&nbsp;	Black–Scholes option pricing

&nbsp;	Discrete delta hedging

&nbsp;	Transaction cost modeling

&nbsp;	Historical market data backtesting

&nbsp;	Geometric Brownian Motion (GBM) Monte Carlo simulation

&nbsp;	Risk metrics (Sharpe, VaR, CVaR)

The objective is to quantify:

&nbsp;	Discrete hedging error

&nbsp;	Transaction cost impact

&nbsp;	Model risk

&nbsp;	Difference between theoretical replication and real-world implementation

This project bridges stochastic calculus theory and practical derivatives implementation.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

&nbsp;Theoretical Background

Under Black–Scholes assumptions, the underlying follows:

dS=μSdt+σSdW



If delta hedging is continuous, replication is exact.

In practice:

&nbsp;	Hedging is discrete

&nbsp;	Volatility is estimated

&nbsp;	Transaction costs exist

&nbsp;	Markets exhibit non-GBM behavior

This framework measures how these frictions impact PnL distribution.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

&nbsp; Architecture

&nbsp;Black–Scholes Engine

&nbsp;	Call price

&nbsp;	Delta

&nbsp;	Gamma

&nbsp;Delta Hedging Engine

&nbsp;	Configurable hedge frequency

&nbsp;	Position sizing

&nbsp;	Transaction cost modeling

&nbsp;	Final PnL computation

&nbsp;Historical Backtest

&nbsp;	Uses yfinance (adjusted prices)

&nbsp;	Rolling realized volatility estimation

&nbsp;	Rolling 30-day short ATM call strategy

&nbsp;GBM Monte Carlo Simulation

&nbsp;	Simulated stochastic paths

&nbsp;	Discrete hedging on synthetic data

&nbsp;	Hedging error distribution analysis

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Outputs

Historical Backtest

&nbsp;	Mean PnL

&nbsp;	Standard deviation

&nbsp;	Sharpe ratio

&nbsp;	95% VaR

&nbsp;	95% CVaR

Monte Carlo Simulation

&nbsp;	Hedging error distribution

&nbsp;	Tail risk metrics

&nbsp;	Histogram visualization





