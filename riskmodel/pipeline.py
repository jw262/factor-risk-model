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


def score_models(comparison):
    # Three tests, one point each: explains the most of the book's daily moves (R²),
    # predicts volatility closest to what actually happened, and leaves the least
    # correlation behind in the leftovers (a model that missed a factor leaves some).
    # Most points wins; a tie goes to the higher R².
    winners = {
        "Explains the most": comparison["R²"].idxmax(),
        "Closest risk forecast": (comparison["predicted vol"]
                                  - comparison["actual vol"]).abs().idxmin(),
        "Cleanest leftovers": comparison["leftover correlation"].idxmin(),
    }
    points = pd.Series(winners).value_counts()
    best = max(points.index, key=lambda name: (points[name], comparison.loc[name, "R²"]))
    return winners, best


def run_analysis(weights, returns, factors, notional, etfs, hedge_pct=1.0):
    # notional = invested capital in $. hedge_pct = how much of it to short in the hedge
    # ETF: 0.5 means short $0.50 of the ETF for every $1 invested.
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
    realized = risk.realized_vol(stock_returns.loc[dates], weights)

    # ---- 1 + 2. Each model's answer to "where do our returns and risk come from?" ----
    # Regression models: regress the whole book on that model's factors.
    by_model = {}
    for name, model, factor_names in [("CAPM", capm, models.CAPM_FACTORS),
                                      ("Fama-French", ff, models.FAMA_FRENCH_FACTORS)]:
        fit = risk.portfolio_regression(portfolio_excess, factors, factor_names)
        by_model[name] = {
            "r2": fit.rsquared,
            "book_betas": fit.params[factor_names].rename(models.FACTOR_LABELS),
            "factor_returns": (factors.loc[portfolio_excess.index, factor_names].mean()
                               * risk.DAYS_PER_YEAR).rename(models.FACTOR_LABELS),
            "attribution": risk.return_attribution(portfolio_excess, factors, factor_names),
            "risk_shares": risk.factor_risk_shares(portfolio_excess, factors, factor_names),
            "stock_betas": model["B"].rename(columns=models.FACTOR_LABELS),
            "stock_stats": model["stats"],
        }
    # Statistical model: split the book into its principal components
    stat_returns = stock_excess[stat["B"].index].dropna() @ stat["B"]
    by_model["Statistical"] = {
        "r2": risk.statistical_r2(weights, stat, stock_returns.loc[dates]),
        "book_betas": stat["B"].T @ weights[stat["B"].index],
        "factor_returns": stat_returns.mean() * risk.DAYS_PER_YEAR,
        "attribution": risk.statistical_attribution(weights, stat, stock_excess),
        "risk_shares": risk.statistical_risk_shares(weights, stat, stock_returns.loc[dates]),
        "stock_betas": stat["B"],
        "stock_stats": None,
    }

    # Then grade all three with the same yardsticks, using each one's Σ = B F Bᵀ + D
    for name, model in [("CAPM", capm), ("Fama-French", ff), ("Statistical", stat)]:
        cov = risk.build_covariance(model)
        n_factors = model["B"].shape[1]
        by_model[name].update({
            "n_factors": n_factors,
            "numbers_estimated": risk.count_numbers_needed(len(tickers), n_factors)[1],
            "positive_definite": risk.is_positive_definite(cov),
            "predicted_vol": risk.portfolio_vol(weights, model)["vol"],
            "leftover_correlation": risk.average_residual_correlation(model["residuals"]),
            "positions": risk.position_risk(weights, cov),
        })

    comparison = pd.DataFrame({name: {
        "factors": m["n_factors"],
        "numbers estimated": m["numbers_estimated"],
        "R²": m["r2"],
        "predicted vol": m["predicted_vol"],
        "actual vol": realized,
        "leftover correlation": m["leftover_correlation"],
        "Σ positive definite": m["positive_definite"],
    } for name, m in by_model.items()}).T.infer_objects()  # give each column a proper type
    winners, best_model = score_models(comparison)

    # ---- 3. Which ETF tracks us best? ----
    etf_returns = returns[list(etfs)].loc[portfolio.index]
    ranking = hedge.rank_hedges(portfolio, etf_returns)
    best = ranking.index[0]
    h = ranking.loc[best, "hedge ratio"]  # the short that would cancel the most risk
    hedged = portfolio - hedge_pct * etf_returns[best]  # the short we actually put on
    spy = returns["SPY"].loc[portfolio.index]

    return {
        "weights": weights,
        "n_stocks": len(tickers),
        "by_model": by_model,
        "comparison": comparison,
        "winners": winners,
        "best_model": best_model,
        "leftover_correlation": comparison["leftover correlation"].to_dict(),
        "n_stat_factors": stat["n_factors"],
        "model_r2": comparison["R²"],
        # The terminal report's headline answers use Fama-French for returns and factor
        # risk, and the statistical model's Σ for positions (it picks up sector factors
        # that Fama-French doesn't have).
        "attribution": by_model["Fama-French"]["attribution"],
        "stock_betas": by_model["Fama-French"]["stock_betas"],
        "stock_stats": by_model["Fama-French"]["stock_stats"],
        "factor_shares": by_model["Fama-French"]["risk_shares"],
        "positions": by_model["Statistical"]["positions"],
        "pca": models.pca_summary(stock_returns.loc[dates]),
        "realized_vol": realized,
        "market_beta": by_model["CAPM"]["book_betas"]["Market"],  # the whole book's CAPM beta
        "portfolio": portfolio,
        "etf_returns": etf_returns,
        "ranking": ranking,
        "best_hedge": best,
        "hedge_ratio": h,
        "hedge_pct": hedge_pct,
        "notional": notional,
        "short_dollars": hedge_pct * notional,
        "beta_before": hedge.beta(portfolio, spy),
        "beta_after": hedge.beta(hedged, spy),
        "vol_after": hedged.std() * risk.DAYS_PER_YEAR ** 0.5,
        "factor_dates": (portfolio_excess.index[0], portfolio_excess.index[-1]),
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
                                    r["notional"], r["hedge_pct"], r["hedge_window"],
                                    path("3_hedge.png")),
    }
