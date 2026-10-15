"""
models.py: the three factor models.

Why factor models at all? You can't build a stock-by-stock covariance matrix
straight from price history. 6,000 stocks = 18 million numbers to estimate and
nowhere near enough data. So we explain returns with a few factors instead.

Every model here hands back the same three pieces:
    B = factor exposures (one row per stock, one column per factor)
    F = factor covariance (how the factors move with each other)
    D = specific variance (the risk that's just that one stock)

risk.py plugs them into  Σ = B F Bᵀ + D  to get the full covariance matrix.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

DAYS_PER_YEAR = 252

CAPM_FACTORS = ["Mkt-RF"]
FAMA_FRENCH_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
FACTOR_LABELS = {
    "Mkt-RF": "Market",
    "SMB": "Size",
    "HML": "Value",
    "RMW": "Profitability",
    "CMA": "Investment",
    "UMD": "Momentum",
}


# ---------------------------------------------------------------------------
# Models 1 and 2: regression models (we pick the factors, the data gives the betas)
# ---------------------------------------------------------------------------

def regression_model(stock_excess, factors, factor_names):
    # One regression per stock:
    #   stock return - risk-free = alpha + beta1*factor1 + beta2*factor2 + ... + noise
    betas = {}
    specific_variance = {}
    residuals = {}
    stats = {}

    for ticker in stock_excess.columns:
        y = stock_excess[ticker].dropna()  # each stock uses all the history it has
        X = sm.add_constant(factors.loc[y.index, factor_names])

        # HAC = Newey-West standard errors. Daily returns are messy, so we don't
        # trust the textbook t-stats.
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

        betas[ticker] = fit.params[factor_names]
        specific_variance[ticker] = fit.resid.var()  # what the factors couldn't explain
        residuals[ticker] = fit.resid
        stats[ticker] = {
            "alpha (per yr)": fit.params["const"] * DAYS_PER_YEAR,
            "alpha t-stat": fit.tvalues["const"],
            "R²": fit.rsquared,
            "days": len(y),
        }

    return {
        "B": pd.DataFrame(betas).T,
        "F": factors[factor_names].cov(),
        "D": pd.Series(specific_variance),
        "residuals": pd.DataFrame(residuals),
        "stats": pd.DataFrame(stats).T,
    }


def capm(stock_excess, factors):
    # Model 1: one factor, the market. Beta = how hard the stock swings when
    # the market moves. AAOI at 3.5 means a 1% market day is a 3.5% day for us.
    return regression_model(stock_excess, factors, CAPM_FACTORS)


def fama_french(stock_excess, factors):
    # Model 2: market + size + value + profitability + investment + momentum.
    # Same regression, more factors. Note the market beta here won't match the
    # CAPM beta: the other factors soak up part of what CAPM called "market."
    return regression_model(stock_excess, factors, FAMA_FRENCH_FACTORS)


# ---------------------------------------------------------------------------
# Model 3: statistical model (the data picks the factors AND the betas)
# ---------------------------------------------------------------------------

def count_real_factors(stock_returns):
    # How many factors are real vs noise? Marchenko-Pastur: if returns were pure
    # noise, no eigenvalue of the correlation matrix would beat (1 + sqrt(N/T))².
    # Anything above that line is real structure.
    T, N = stock_returns.shape
    eigenvalues = np.linalg.eigvalsh(stock_returns.corr())
    noise_edge = (1 + np.sqrt(N / T)) ** 2
    return max(1, int((eigenvalues > noise_edge).sum()))


def statistical_model(stock_returns, n_factors=None):
    # No pre-picked factors here. We let the data tell us what moves together.
    returns = stock_returns.dropna()  # PCA needs every stock on every day
    S = returns.cov()

    # Eigendecomposition: S = Q Λ Qᵀ
    # Each eigenvector = a "factor portfolio". Each eigenvalue = how much variance it carries.
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    order = np.argsort(eigenvalues)[::-1]  # biggest first
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    # Eigenvectors can come out flipped (the sign doesn't matter mathematically).
    # Make each one mostly positive so the charts read right.
    eigenvectors = eigenvectors * np.where(eigenvectors.sum(axis=0) < 0, -1, 1)

    if n_factors is None:
        n_factors = count_real_factors(returns)
    names = [f"PC{k + 1}" for k in range(n_factors)]

    # Keep the top K factors, toss the rest as noise
    B = pd.DataFrame(eigenvectors[:, :n_factors], index=S.index, columns=names)
    F = pd.DataFrame(np.diag(eigenvalues[:n_factors]), index=names, columns=names)

    # Whatever variance the top K factors don't cover becomes specific risk
    covered = np.diag(B.values @ F.values @ B.values.T)
    D = pd.Series(np.diag(S) - covered, index=S.index).clip(lower=1e-12)

    return {"B": B, "F": F, "D": D, "n_factors": n_factors}


def pca_summary(stock_returns):
    # For the concentration chart: how much each PC explains, and how many
    # truly different bets we're holding.
    returns = stock_returns.dropna()
    T, N = returns.shape
    eigenvalues = np.sort(np.linalg.eigvalsh(returns.corr()))[::-1]

    # Effective bets: 12 stocks that move independently = 12. All moving as one = 1.
    effective_bets = eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()

    return {
        "explained": pd.Series(eigenvalues / N, index=[f"PC{k + 1}" for k in range(N)]),
        "noise_edge": (1 + np.sqrt(N / T)) ** 2,
        "effective_bets": effective_bets,
        "n_stocks": N,
    }
