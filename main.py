"""
main.py: run the whole thing from the terminal.

    python main.py

Three questions, in order:
    1. Where are our returns coming from?
    2. Where is our risk coming from, and are we concentrated in one factor?
    3. Which ETF tracks our book the best, and how much do we short to go market neutral?

Charts and tables land in the reports/ folder. (Want the website version? python -m streamlit run app.py)
"""
from pathlib import Path

import pandas as pd

from riskmodel import data, hedge, pipeline, risk

# ---- Settings: change these ----
PORTFOLIO_FILE = "portfolio.csv"
START_DATE = "2023-01-01"
NOTIONAL = 17_800  # $ size of the book, used to size the hedge
OUTPUT = Path("reports")


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def as_percent(series, decimals=0):
    template = "{:." + str(decimals) + "%}"  # e.g. "{:.1%}" turns 0.256 into 25.6%
    return series.map(template.format).to_string()


def main():
    OUTPUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 140)

    # ---- Step 1: Get the data ----
    weights = data.load_portfolio(PORTFOLIO_FILE)
    tickers = pipeline.tickers_to_download(weights, hedge.CANDIDATE_ETFS)
    print(f"Downloading prices for {len(tickers)} tickers...")
    returns = data.to_returns(data.load_prices(tickers, START_DATE))
    print("Downloading Fama-French factors...")
    factors = data.load_factors()

    # ---- Step 2: Run everything ----
    r = pipeline.run_analysis(weights, returns, factors, NOTIONAL, hedge.CANDIDATE_ETFS)

    # ---- Step 3: Print the story ----
    section("WHY FACTOR MODELS")
    direct, factor_model = risk.count_numbers_needed(6000, 6)
    print(f"Full covariance matrix for 6,000 stocks: {direct:,} numbers to estimate")
    print(f"6-factor model for 6,000 stocks:         {factor_model:,} numbers")
    print(f"That's {direct / factor_model:,.0f}x fewer, and no need for 24 years of daily data.")

    section("THE THREE MODELS")
    table = r["comparison"].copy()
    for column in ["predicted vol", "actual vol"]:
        table[column] = table[column].map("{:.1%}".format)
    print(table.to_string())
    for name, value in r["leftover_correlation"].items():
        print(f"{name}: average correlation of leftovers = {value:.2f} (model assumes 0)")
    print(f"Statistical: kept {r['n_stat_factors']} factor(s) above the noise line")

    section("1. WHERE ARE OUR RETURNS COMING FROM?")
    print(f"Book's market beta (CAPM): {r['market_beta']:.2f}\n")
    print("Share of daily moves explained (R²):")
    print(as_percent(r["model_r2"]))
    print("\nWhere the yearly excess return came from:")
    print(r["attribution"].map("{:+.1%}".format).to_string())
    print(f"Total: {r['attribution'].sum():+.1%}")

    section("2. WHERE IS OUR RISK COMING FROM?")
    shares = r["factor_shares"]
    print("Share of our risk by factor (Fama-French):")
    print(as_percent(shares, 1))
    top = shares.drop("Specific").idxmax()
    if shares[top] > 0.5:
        print(f"-> CONCENTRATED: {shares[top]:.0%} of our risk is {top}")
    print(f"\nEffective independent bets: {r['pca']['effective_bets']:.1f} out of "
          f"{r['n_stocks']} names")
    print("\nMoney vs risk by position:")
    print(r["positions"].map("{:.1%}".format).to_string())

    section("3. WHICH ETF TRACKS OUR BOOK THE BEST?")
    ranking = r["ranking"].copy()
    for column in ["R²", "vol before", "vol after"]:
        ranking[column] = ranking[column].map("{:.0%}".format)
    print(ranking.to_string(float_format="{:.2f}".format))
    print(f"\nBest match: {r['best_hedge']}. Short {r['hedge_ratio']:.2f}x = "
          f"${r['short_dollars']:,.0f} against our ${NOTIONAL:,.0f}")
    print(f"Beta to SPY: {r['beta_before']:.2f} before -> {r['beta_after']:.2f} after")

    # ---- Step 4: Save charts and tables ----
    pipeline.make_charts(r, OUTPUT)
    r["comparison"].to_csv(OUTPUT / "model_comparison.csv")
    pd.concat([r["stock_betas"], r["stock_stats"]], axis=1).to_csv(OUTPUT / "stock_factor_betas.csv")
    r["attribution"].to_csv(OUTPUT / "return_attribution.csv")
    r["factor_shares"].to_csv(OUTPUT / "risk_by_factor.csv")
    r["positions"].to_csv(OUTPUT / "risk_by_position.csv")
    r["ranking"].to_csv(OUTPUT / "hedge_ranking.csv")
    print(f"\nDone. Charts and tables are in {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
