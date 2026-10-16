"""
risk.py: turning a factor model into actual risk numbers.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .models import FACTOR_LABELS

DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# The covariance matrix
# ---------------------------------------------------------------------------

def build_covariance(model):
    # The money formula: Σ = B F Bᵀ + D
    # Two different stocks only move together because they share factors:
    #   Cov(stock i, stock j) = beta_i' F beta_j
    B, F, D = model["B"], model["F"], model["D"]
    cov = B.values @ F.values @ B.values.T + np.diag(D.values)
    return pd.DataFrame(cov, index=B.index, columns=B.index)


def is_positive_definite(cov):
    # Cholesky only works if every eigenvalue is > 0. If it runs, the matrix is legit:
    # every portfolio has positive risk and we can invert it.
    # Why it's guaranteed here: xᵀΣx = (Bᵀx)ᵀ F (Bᵀx) + Σ d_i x_i². First part is >= 0
    # (F is a covariance matrix), second part is > 0 (every d_i > 0). So the total is > 0.
    try:
        np.linalg.cholesky(cov)
        return True
    except np.linalg.LinAlgError:
        return False


def count_numbers_needed(n_stocks, n_factors):
    # The data problem in one function.
    #   Direct: every variance + every pair of stocks = N(N+1)/2
    #   Factor model: betas (N×K) + factor covariance + one specific risk per stock
    direct = n_stocks * (n_stocks + 1) // 2
    factor_model = n_stocks * n_factors + n_factors * (n_factors + 1) // 2 + n_stocks
    return direct, factor_model


# ---------------------------------------------------------------------------
# Portfolio risk
# ---------------------------------------------------------------------------

def portfolio_vol(weights, model):
    # Split the book's risk into factor-driven vs stock-specific.
    #   b = Bᵀw is the whole book's exposure to each factor
    #   factor variance   = bᵀ F b
    #   specific variance = Σ w² × d   (shrinks as we add names... if they're truly unrelated)
    B, F, D = model["B"], model["F"], model["D"]
    w = weights[B.index]
    b = B.T @ w
    factor_var = (b @ F @ b) * DAYS_PER_YEAR
    specific_var = (w ** 2 * D).sum() * DAYS_PER_YEAR
    total_var = factor_var + specific_var
    return {
        "vol": np.sqrt(total_var),
        "factor share": factor_var / total_var,
        "specific share": specific_var / total_var,
    }


def realized_vol(stock_returns, weights):
    # What actually happened. This is the scoreboard each model gets graded against.
    portfolio = stock_returns[weights.index].dropna() @ weights
    return portfolio.std() * np.sqrt(DAYS_PER_YEAR)


def average_residual_correlation(residuals):
    # Factor models assume leftovers don't move together (correlation = 0).
    # If they do, the model is missing a factor. For a semis book, it's probably the sector.
    # We average the SIZE of each correlation: +0.3 and -0.3 shouldn't cancel out to "clean".
    corr = residuals.corr().values
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
    return np.nanmean(np.abs(off_diagonal))


def position_risk(weights, cov):
    # Euler risk contribution: RC_i = w_i × (Σw)_i / wᵀΣw. These add up to 100%.
    # If a stock's risk share is bigger than its weight, it's punching above its size.
    w = weights[cov.index]
    marginal = cov @ w
    contribution = w * marginal / (w @ marginal)
    table = pd.DataFrame({"weight": w, "risk share": contribution})
    return table.sort_values("risk share", ascending=False)


# ---------------------------------------------------------------------------
# Portfolio-level attribution: regress the whole book on the factors
# ---------------------------------------------------------------------------

def portfolio_regression(portfolio_excess, factors, factor_names):
    X = sm.add_constant(factors.loc[portfolio_excess.index, factor_names])
    return sm.OLS(portfolio_excess, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})


def return_attribution(portfolio_excess, factors, factor_names):
    # Where the return came from: beta × average factor return, for each factor, plus alpha.
    # This adds up EXACTLY to our average return, because a regression with a
    # constant forces the leftover noise to average out to zero.
    fit = portfolio_regression(portfolio_excess, factors, factor_names)
    avg_factor_return = factors.loc[portfolio_excess.index, factor_names].mean()
    contributions = fit.params[factor_names] * avg_factor_return * DAYS_PER_YEAR
    contributions["Alpha"] = fit.params["const"] * DAYS_PER_YEAR
    return contributions.rename(index=FACTOR_LABELS)


def factor_risk_shares(portfolio_excess, factors, factor_names):
    # Where the risk comes from: Var(book) = bᵀ F b + Var(leftover).
    # Each factor's piece is b_k × (F b)_k. All the pieces add up to 100%.
    fit = portfolio_regression(portfolio_excess, factors, factor_names)
    b = fit.params[factor_names]
    F = factors.loc[portfolio_excess.index, factor_names].cov()
    total = portfolio_excess.var()
    shares = b * (F @ b) / total
    shares["Specific"] = fit.resid.var() / total
    return shares.rename(index=FACTOR_LABELS)


def statistical_r2(weights, stat_model, stock_returns):
    # The PCA version of R²: what share of the book's variance the top K factors capture.
    S = stock_returns.dropna().cov()
    w = weights[S.index]
    b = stat_model["B"].T @ w
    return (b @ stat_model["F"] @ b) / (w @ S @ w)


def statistical_attribution(weights, stat_model, stock_excess):
    # The PCA version of return attribution. Each factor is a portfolio (its eigenvector),
    # so its daily return is f = Bᵀr. The book splits into  r_p = bᵀf + leftover  with b = Bᵀw.
    # Averaging both sides: the pieces add up exactly to the book's average return.
    B = stat_model["B"]
    returns = stock_excess[B.index].dropna()
    w = weights[B.index]
    b = B.T @ w
    f = returns @ B
    contributions = b * f.mean() * DAYS_PER_YEAR
    contributions["Alpha"] = (returns @ w - f @ b).mean() * DAYS_PER_YEAR
    return contributions


def statistical_risk_shares(weights, stat_model, stock_returns):
    # The PCA version of risk attribution: Var(book) = Σ b_k² λ_k + Var(leftover).
    # The PCs are uncorrelated with each other and with the leftover, so the pieces add
    # up to 100%, and the factor pieces together equal statistical_r2.
    B = stat_model["B"]
    returns = stock_returns[B.index].dropna()
    w = weights[B.index]
    b = B.T @ w
    total = (returns @ w).var()
    eigenvalues = pd.Series(np.diag(stat_model["F"]), index=b.index)
    shares = b ** 2 * eigenvalues / total
    shares["Specific"] = (returns @ w - returns @ B @ b).var() / total
    return shares
