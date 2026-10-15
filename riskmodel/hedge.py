"""
hedge.py: find the ETF that tracks our book the best and figure out how much to short.
"""
import numpy as np
import pandas as pd

DAYS_PER_YEAR = 252

# The lineup. SMH is dominated by NVDA, TSM and AVGO (none of which we own),
# so an equal-weight semis ETF like XSD might actually look more like us.
CANDIDATE_ETFS = ["SPY", "QQQ", "XLK", "SMH", "SOXX", "PSI", "XSD", "IGV", "IWM"]


def hedge_ratio(portfolio, etf):
    # Minimum-variance hedge: h* = Cov(book, ETF) / Var(ETF)
    # Short h* dollars of the ETF for every $1 we're long. Same number as the regression slope.
    return portfolio.cov(etf) / etf.var()


def rank_hedges(portfolio, etf_returns):
    # Score every ETF. Higher correlation = tracks us better = kills more risk.
    # After hedging, the variance left over is Var(book) × (1 - correlation²).
    rows = {}
    for etf in etf_returns.columns:
        h = hedge_ratio(portfolio, etf_returns[etf])
        hedged = portfolio - h * etf_returns[etf]
        correlation = portfolio.corr(etf_returns[etf])
        rows[etf] = {
            "correlation": correlation,
            "R²": correlation ** 2,
            "hedge ratio": h,
            "vol before": portfolio.std() * np.sqrt(DAYS_PER_YEAR),
            "vol after": hedged.std() * np.sqrt(DAYS_PER_YEAR),
        }
    return pd.DataFrame(rows).T.sort_values("R²", ascending=False)


def beta(returns, market):
    # Same formula as the hedge ratio. Against SPY: ~0 means we're market neutral.
    return returns.cov(market) / market.var()
