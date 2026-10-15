"""
pipeline.py: the brain.

Takes a book + the data, runs every model, hands back all the results.
main.py (terminal) and app.py (website) both call this, so the math lives in one place.
"""
import pandas as pd

from . import charts, hedge, models, risk

MIN_DAYS = 120  # need about 6 months where every stock trades, or the estimates are junk


def tickers_to_download(weights, etfs):
    # Our stocks + the hedge candidates + SPY (always needed for the market-neutral check).
    # dict.fromkeys is a quick way to drop duplicates and keep the order.
    return list(dict.fromkeys(list(weights.index) + list(etfs) + ["SPY"]))


def date_range(series):
    return f"{series.index[0]:%b %d, %Y} to {series.index[-1]:%b %d, %Y}"


def run_analysis(weights, returns, factors, notional, etfs):
    tickers = list(weights.index)
    stock_returns = returns[tickers]

    # French's data runs a month or two behind, so the factor models stop where it stops
    dates = stock_returns.index.intersection(factors.index)
    factors = factors.loc[dates]
    stock_excess = stock_returns.loc[dates].sub(factors["RF"], axis=0)  # return minus T-bills

    # Our book's daily return = weighted sum of the stocks (today's weights, held constant)
    portfolio = stock_returns.dropna() @ weights
    portfolio_excess = stock_excess.dropna() @ weights
    if len(portfolio_excess) < MIN_DAYS:
        raise ValueError(
            f"Only {len(portfolio_excess)} days where every stock has data; at least "
            f"{MIN_DAYS} are needed. A recent IPO or spin-off in the list is the usual cause.")

    # ---- The three models. Each one gives us B, F, D. ----
    capm = models.capm(stock_excess, factors)
    ff = models.fama_french(stock_excess, factors)
    stat = models.statistical_model(stock_returns.loc[dates])
    all_models = {"CAPM": capm, "Fama-French": ff, "Statistical": stat}
    realized = risk.realized_vol(stock_returns.loc[dates], weights)

    comparison = {}
    for name, model in all_models.items():
        cov = risk.build_covariance(model)  # Σ = B F Bᵀ + D
        n_factors = model["B"].shape[1]
        comparison[name] = {
            "factors": n_factors,
            "numbers estimated": risk.count_numbers_needed(len(tickers), n_factors)[1],
            "predicted vol": risk.portfolio_vol(weights, model)["vol"],
            "actual vol": realized,
            "Σ positive definite": risk.is_positive_definite(cov),
        }
    comparison = pd.DataFrame(comparison).T.infer_objects()  # give each column a proper type

    # ---- 1. Where are our returns coming from? ----
    capm_fit = risk.portfolio_regression(portfolio_excess, factors, models.CAPM_FACTORS)
    model_r2 = pd.Series({
        "CAPM": capm_fit.rsquared,
        "Fama-French": risk.portfolio_regression(portfolio_excess, factors,
                                                 models.FAMA_FRENCH_FACTORS).rsquared,
        "Statistical": risk.statistical_r2(weights, stat, stock_returns.loc[dates]),
    })
    attribution = risk.return_attribution(portfolio_excess, factors,
                                          models.FAMA_FRENCH_FACTORS)

    # ---- 2. Where is our risk coming from? ----
    factor_shares = risk.factor_risk_shares(portfolio_excess, factors,
                                            models.FAMA_FRENCH_FACTORS)
    # Position risk uses the statistical model's Σ: it's the one that picks up
    # sector factors that Fama-French doesn't have.
    positions = risk.position_risk(weights, risk.build_covariance(stat))

    # ---- 3. Which ETF tracks us best? ----
    etf_returns = returns[list(etfs)].loc[portfolio.index]
    ranking = hedge.rank_hedges(portfolio, etf_returns)
    best = ranking.index[0]
    h = ranking.loc[best, "hedge ratio"]
    hedged = portfolio - h * etf_returns[best]
    spy = returns["SPY"].loc[portfolio.index]

    return {
        "weights": weights,
        "n_stocks": len(tickers),
        "comparison": comparison,
        "leftover_correlation": {name: risk.average_residual_correlation(m["residuals"])
                                 for name, m in [("CAPM", capm), ("Fama-French", ff)]},
        "n_stat_factors": stat["n_factors"],
        "model_r2": model_r2,
        "attribution": attribution,
        "stock_betas": ff["B"].rename(columns=models.FACTOR_LABELS),
        "stock_stats": ff["stats"],
        "factor_shares": factor_shares,
        "pca": models.pca_summary(stock_returns.loc[dates]),
        "positions": positions,
        "realized_vol": realized,
        "market_beta": capm_fit.params["Mkt-RF"],  # the whole book's CAPM beta
        "portfolio": portfolio,
        "etf_returns": etf_returns,
        "ranking": ranking,
        "best_hedge": best,
        "hedge_ratio": h,
        "notional": notional,
        "short_dollars": h * notional,
        "beta_before": hedge.beta(portfolio, spy),
        "beta_after": hedge.beta(hedged, spy),
        "vol_after": hedged.std() * risk.DAYS_PER_YEAR ** 0.5,
        "factor_window": "Factor models: " + date_range(portfolio_excess),
        "hedge_window": "Hedge analysis: " + date_range(portfolio),
    }


def make_charts(results, folder=None):
    # The three charts. With a folder they get saved as PNGs; without one they come back as figures.
    def path(name):
        return folder / name if folder else None

    r = results
    return {
        "returns": charts.returns_chart(r["model_r2"], r["attribution"], r["stock_betas"],
                                        r["factor_window"], path("1_returns.png")),
        "risk": charts.risk_chart(r["factor_shares"], r["pca"], r["positions"],
                                  r["comparison"]["predicted vol"].astype(float),
                                  r["realized_vol"], r["factor_window"], path("2_risk.png")),
        "hedge": charts.hedge_chart(r["portfolio"], r["etf_returns"], r["ranking"],
                                    r["notional"], r["hedge_window"], path("3_hedge.png")),
    }
