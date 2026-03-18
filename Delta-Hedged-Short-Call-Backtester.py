# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

# ==========================================================
# CONFIGURATION
# ==========================================================

CONFIG = {
    "ticker": "AAPL",
    "start_date": "2022-01-01",
    "end_date": "2023-12-31",
    "r": 0.03,
    "maturity_days": 30,
    "transaction_cost": 0.0005,
    "position_size": 1,          # number of options shorted
    "hedge_frequency": 1,        # rebalance every 1 day
    "rolling_vol_window": 20,
    "gbm_mu": 0.08,
    "gbm_sigma": 0.25,
    "monte_carlo_paths": 1000,
    "seed": 42
}

TRADING_DAYS = 252


# ==========================================================
# BLACK-SCHOLES FUNCTIONS
# ==========================================================

def bs_price(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S - K * np.exp(-r * T), 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta(S, K, T, r, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    if sigma <= 0:
        return 1.0 if S > K * np.exp(-r * T) else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


# ==========================================================
# HISTORICAL DATA LOADER
# ==========================================================

def load_historical_data(config):
    df = yf.download(
        config["ticker"],
        start=config["start_date"],
        end=config["end_date"],
        auto_adjust=True,
        progress=False
    )

    if "Close" not in df.columns:
        raise ValueError("Close column not found in downloaded data")

    df = df[["Close"]].rename(columns={"Close": "price"})
    df["log_ret"] = np.log(df["price"] / df["price"].shift(1))

    df["vol"] = (
        df["log_ret"]
        .rolling(config["rolling_vol_window"])
        .std()
        * np.sqrt(TRADING_DAYS)
    )

    df = df.dropna().copy()
    return df


# ==========================================================
# GBM SIMULATION
# ==========================================================

def simulate_gbm(S0, mu, sigma, T_days, paths, seed=None):
    if seed is not None:
        np.random.seed(seed)

    dt = 1 / TRADING_DAYS
    prices = np.zeros((T_days + 1, paths))
    prices[0, :] = S0

    for t in range(1, T_days + 1):
        z = np.random.normal(size=paths)
        prices[t, :] = prices[t - 1, :] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )

    return prices


# ==========================================================
# DELTA HEDGING ENGINE
# ==========================================================

def delta_hedge_path(price_path, sigma, config):
    maturity = config["maturity_days"]
    r = config["r"]
    tc = config["transaction_cost"]
    position = config["position_size"]
    hedge_freq = config["hedge_frequency"]

    S0 = float(price_path[0])
    K = S0
    T0 = maturity / TRADING_DAYS

    # Assume short call: receive premium at t=0
    option_premium = bs_price(S0, K, T0, r, sigma) * position
    cash = option_premium
    hedge_position = 0.0

    for t in range(len(price_path)):
        S = float(price_path[t])
        T = max((maturity - t) / TRADING_DAYS, 0)

        # cash earns risk-free interest after day 0
        if t > 0:
            cash *= np.exp(r / TRADING_DAYS)

        # rebalance hedge only before expiry
        if (t % hedge_freq == 0) and (T > 0):
            delta = bs_delta(S, K, T, r, sigma) * position
            target_hedge = -delta
            hedge_change = target_hedge - hedge_position

            transaction_cost = abs(hedge_change) * S * tc

            cash -= hedge_change * S
            cash -= transaction_cost

            hedge_position = target_hedge

    # settle at expiry
    S_T = float(price_path[-1])
    option_payoff = max(S_T - K, 0.0) * position

    # close hedge at expiry
    cash += hedge_position * S_T
    cash -= abs(hedge_position) * S_T * tc

    final_pnl = cash - option_payoff
    return final_pnl


# ==========================================================
# HISTORICAL BACKTEST
# ==========================================================

def run_historical_backtest(data, config):
    pnl_results = []
    maturity_days = config["maturity_days"]

    # Need maturity_days + 1 prices for a full path
    for i in range(len(data) - maturity_days):
        window = data.iloc[i : i + maturity_days + 1]

        if len(window) < maturity_days + 1:
            continue

        prices = window["price"].values
        sigma = float(window["vol"].iloc[0])

        pnl = delta_hedge_path(prices, sigma, config)
        pnl_results.append(pnl)

    return np.array(pnl_results)


# ==========================================================
# MONTE CARLO BACKTEST
# ==========================================================

def run_monte_carlo(config):
    S0 = 100.0

    prices = simulate_gbm(
        S0=S0,
        mu=config["gbm_mu"],
        sigma=config["gbm_sigma"],
        T_days=config["maturity_days"],
        paths=config["monte_carlo_paths"],
        seed=config["seed"]
    )

    hedging_errors = []

    for i in range(config["monte_carlo_paths"]):
        path = prices[:, i]
        pnl = delta_hedge_path(path, config["gbm_sigma"], config)
        hedging_errors.append(pnl)

    return np.array(hedging_errors)


# ==========================================================
# RISK METRICS
# ==========================================================

def risk_metrics(pnl_array):
    mean = np.mean(pnl_array)
    std = np.std(pnl_array, ddof=1) if len(pnl_array) > 1 else 0.0
    sharpe = mean / std if std != 0 else 0.0
    var_95 = np.percentile(pnl_array, 5)
    cvar_95 = pnl_array[pnl_array <= var_95].mean()

    return {
        "Mean PnL": mean,
        "Std Dev": std,
        "Sharpe": sharpe,
        "VaR 95%": var_95,
        "CVaR 95%": cvar_95
    }


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("Running Historical Backtest...")
    hist_data = load_historical_data(CONFIG)
    hist_pnl = run_historical_backtest(hist_data, CONFIG)
    hist_stats = risk_metrics(hist_pnl)

    print("\nHistorical Stats:")
    for k, v in hist_stats.items():
        print(f"{k}: {v:.4f}")

    print("\nRunning GBM Monte Carlo Simulation...")
    mc_pnl = run_monte_carlo(CONFIG)
    mc_stats = risk_metrics(mc_pnl)

    print("\nMonte Carlo Stats:")
    for k, v in mc_stats.items():
        print(f"{k}: {v:.4f}")

    # Plot historical hedging PnL
    plt.figure(figsize=(10, 5))
    plt.hist(hist_pnl, bins=40)
    plt.title("Historical Delta-Hedging PnL Distribution")
    plt.xlabel("PnL")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.show()

    # Plot Monte Carlo hedging PnL
    plt.figure(figsize=(10, 5))
    plt.hist(mc_pnl, bins=40)
    plt.title("GBM Monte Carlo Hedging Error Distribution")
    plt.xlabel("PnL")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.show()