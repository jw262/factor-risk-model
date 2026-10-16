"""
test_math.py: proving the math claims in the README with code.

Run with:  python -m pytest
Every test builds fake data where we KNOW the right answer, then checks the code gets it.
"""
import numpy as np
import pandas as pd

from riskmodel import data, hedge, models, risk

rng = np.random.default_rng(42)


def fake_market_data(days=1500):
    # A fake market plus three fake stocks with betas we pick on purpose
    dates = pd.bdate_range("2020-01-01", periods=days)
    factors = pd.DataFrame({
        "Mkt-RF": rng.normal(0.0004, 0.010, days),
        "SMB": rng.normal(0, 0.005, days),
        "HML": rng.normal(0, 0.005, days),
        "RMW": rng.normal(0, 0.004, days),
        "CMA": rng.normal(0, 0.004, days),
        "UMD": rng.normal(0, 0.006, days),
        "RF": 0.0001,
    }, index=dates)
    true_betas = {"AAA": 0.8, "BBB": 1.5, "CCC": 2.5}
    stocks = pd.DataFrame({ticker: b * factors["Mkt-RF"] + rng.normal(0, 0.015, days)
                           for ticker, b in true_betas.items()}, index=dates)
    return stocks, factors, true_betas


# --- The data problem -------------------------------------------------------

def test_direct_covariance_breaks_with_more_stocks_than_days():
    # THE reason this project exists. 100 stocks, only 60 days of history:
    # the sample covariance matrix is singular (some eigenvalues are 0) and
    # Cholesky fails. You can't trust it, and you can't invert it.
    returns = pd.DataFrame(rng.normal(0, 0.02, size=(60, 100)))
    sample_cov = returns.cov()
    assert not risk.is_positive_definite(sample_cov)
    assert np.linalg.eigvalsh(sample_cov).min() < 1e-12


def test_factor_model_needs_way_fewer_numbers():
    direct, factor_model = risk.count_numbers_needed(6000, 6)
    assert direct == 18_003_000
    assert factor_model == 42_021


def test_factor_covariance_is_always_positive_definite():
    # Same 100 stocks, same 60 days, but built as Σ = B F Bᵀ + D: always invertible.
    B = pd.DataFrame(rng.normal(1, 0.5, size=(100, 3)))
    F = pd.DataFrame(np.cov(rng.normal(size=(3, 60))))
    D = pd.Series(rng.uniform(1e-4, 4e-4, 100))
    cov = risk.build_covariance({"B": B, "F": F, "D": D})
    assert risk.is_positive_definite(cov)


# --- The models ---------------------------------------------------------------

def test_capm_finds_the_betas_we_planted():
    stocks, factors, true_betas = fake_market_data()
    excess = stocks.sub(factors["RF"], axis=0)
    capm = models.capm(excess, factors)
    for ticker, b in true_betas.items():
        assert abs(capm["B"].loc[ticker, "Mkt-RF"] - b) < 0.1


def test_statistical_model_is_positive_definite():
    stocks, _, _ = fake_market_data()
    stat = models.statistical_model(stocks)
    assert stat["n_factors"] >= 1
    assert (stat["D"] > 0).all()
    assert risk.is_positive_definite(risk.build_covariance(stat))


# --- Attribution ------------------------------------------------------------

def test_return_attribution_adds_up_exactly():
    stocks, factors, _ = fake_market_data()
    weights = pd.Series({"AAA": 0.2, "BBB": 0.3, "CCC": 0.5})
    book = stocks.sub(factors["RF"], axis=0) @ weights
    pieces = risk.return_attribution(book, factors, models.FAMA_FRENCH_FACTORS)
    assert np.isclose(pieces.sum(), book.mean() * 252)


def test_risk_shares_add_to_100_percent():
    stocks, factors, _ = fake_market_data()
    weights = pd.Series({"AAA": 0.2, "BBB": 0.3, "CCC": 0.5})
    book = stocks.sub(factors["RF"], axis=0) @ weights
    shares = risk.factor_risk_shares(book, factors, models.FAMA_FRENCH_FACTORS)
    assert np.isclose(shares.sum(), 1.0)


def test_statistical_attribution_adds_up_exactly():
    # Same two promises for the PCA model: return pieces add up to the book's
    # return, and risk pieces add up to 100% (PCs are uncorrelated with the leftover).
    stocks, factors, _ = fake_market_data()
    weights = pd.Series({"AAA": 0.2, "BBB": 0.3, "CCC": 0.5})
    excess = stocks.sub(factors["RF"], axis=0)
    stat = models.statistical_model(stocks)

    pieces = risk.statistical_attribution(weights, stat, excess)
    assert np.isclose(pieces.sum(), (excess @ weights).mean() * 252)

    shares = risk.statistical_risk_shares(weights, stat, stocks)
    assert np.isclose(shares.sum(), 1.0)
    assert np.isclose(shares.drop("Specific").sum(), risk.statistical_r2(weights, stat, stocks))


# --- The hedge ----------------------------------------------------------------

def test_hedge_ratio_is_the_lowest_risk_choice():
    stocks, factors, _ = fake_market_data()
    book = stocks.mean(axis=1)
    etf = factors["Mkt-RF"]
    h = hedge.hedge_ratio(book, etf)

    def risk_if_we_short(amount):
        return (book - amount * etf).var()

    # Shorting a little more or a little less than h* both leave MORE risk
    assert risk_if_we_short(h) < risk_if_we_short(h + 0.05)
    assert risk_if_we_short(h) < risk_if_we_short(h - 0.05)


def test_hedged_variance_is_one_minus_correlation_squared():
    stocks, factors, _ = fake_market_data()
    book = stocks.mean(axis=1)
    etf = factors["Mkt-RF"]
    hedged = book - hedge.hedge_ratio(book, etf) * etf
    assert np.isclose(hedged.var(), book.var() * (1 - book.corr(etf) ** 2))


# --- Data -------------------------------------------------------------------

def test_reads_french_file_format():
    text = (
        "This file was created using the 202608 CRSP database.\n"
        "\n"
        ",Mkt-RF,SMB,HML,RF\n"
        "20240102,  -0.71,  -0.02,   1.02,  0.021\n"
        "20240103,  -1.04,  -0.17,   0.25,  0.021\n"
        "\n"
        "Copyright 2026 Eugene F. Fama and Kenneth R. French\n"
    )
    table = data.parse_french_text(text)
    assert list(table.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(table) == 2
    assert np.isclose(table.iloc[0]["Mkt-RF"], -0.0071)  # -0.71% -> -0.0071


# --- The whole pipeline -------------------------------------------------------

def test_full_pipeline_runs_and_picks_the_right_hedge():
    # Fake world: three stocks that all ride a hidden "sector" factor, and a fake
    # sector ETF that tracks it. The pipeline should pick that ETF as the best hedge.
    from riskmodel import pipeline

    stocks, factors, _ = fake_market_data()
    sector = pd.Series(rng.normal(0, 0.012, len(stocks)), index=stocks.index)
    stocks = stocks.add(sector, axis=0)
    returns = stocks.copy()
    returns["SPY"] = factors["Mkt-RF"] + factors["RF"]
    returns["SECTOR_ETF"] = 1.3 * factors["Mkt-RF"] + sector + rng.normal(0, 0.003, len(stocks))
    weights = pd.Series({"AAA": 0.2, "BBB": 0.3, "CCC": 0.5})

    results = pipeline.run_analysis(weights, returns, factors, notional=10_000,
                                    etfs=["SPY", "SECTOR_ETF"])

    assert results["best_hedge"] == "SECTOR_ETF"
    assert results["comparison"]["Σ positive definite"].all()
    assert abs(results["beta_after"]) < abs(results["beta_before"])
