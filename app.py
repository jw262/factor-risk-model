"""
app.py: the website version.

    python -m streamlit run app.py

Anyone types in their stocks and weights, hits Run, and gets the same three-part
report as main.py. All the math comes from riskmodel/pipeline.py, same as the terminal.
"""
from datetime import date

import pandas as pd
import streamlit as st

from riskmodel import data, hedge, pipeline, risk

st.set_page_config(page_title="Factor Risk Model", page_icon="📉", layout="wide")

MAX_STOCKS = 50


# ---------------------------------------------------------------------------
# Downloads are slow, so we remember them (prices for an hour, factors for a day)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_returns(tickers, start_date):
    return data.to_returns(data.load_prices(list(tickers), start_date))


@st.cache_data(ttl=86400, show_spinner=False)
def get_factors():
    return data.load_factors()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_holdings(table):
    # Tidy up whatever got typed in: uppercase tickers, skip blank rows, merge duplicates
    table = table.dropna(subset=["ticker", "weight"]).copy()
    table["ticker"] = table["ticker"].astype(str).str.strip().str.upper()
    table = table[table["ticker"] != ""]
    return table.groupby("ticker")["weight"].sum()


def as_percent(table, columns, decimals=0):
    template = "{:." + str(decimals) + "%}"
    table = table.copy()
    for column in columns:
        table[column] = table[column].map(template.format)
    return table


def stop_with(message):
    st.error(message)
    st.stop()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

st.title("Factor risk model")
st.write("Enter your holdings to see where your returns and risk come from, "
         "and which ETF tracks your book closely enough to hedge it.")

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Your holdings")
    holdings = st.data_editor(
        pd.read_csv("portfolio.csv"),
        num_rows="dynamic",
        hide_index=True,
        key="holdings",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", required=True),
            "weight": st.column_config.NumberColumn(
                "Weight", min_value=0.0, format="%.4f",
                help="Any scale works. 25 or 0.25 both become 25% once the weights are added up."),
        },
    )
    st.caption("Add rows at the bottom of the table. Long positions only for now.")

with right:
    st.subheader("Settings")
    start = st.date_input("History starts", value=date(2023, 1, 1),
                          min_value=date(2005, 1, 1), max_value=date.today())
    notional = st.number_input("Book size ($)", min_value=1_000, value=17_800, step=1_000,
                               help="Used to turn the hedge ratio into a dollar amount to short.")
    etfs = st.multiselect("Hedge candidates", options=hedge.CANDIDATE_ETFS,
                          default=hedge.CANDIDATE_ETFS)
    extra = st.text_input("Other ETFs to test (optional)", placeholder="e.g. ARKK, XLE")
    run = st.button("Run analysis", type="primary", width="stretch")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if run:
    weights = clean_holdings(holdings)
    if len(weights) < 2:
        stop_with("Add at least 2 stocks to run the models.")
    if len(weights) > MAX_STOCKS:
        stop_with(f"Keep it to {MAX_STOCKS} stocks or fewer.")
    if (weights <= 0).any():
        stop_with("Every weight needs to be above 0. Shorts aren't supported yet.")

    etf_list = list(etfs) + [e.strip().upper() for e in extra.split(",") if e.strip()]
    etf_list = list(dict.fromkeys(etf_list))
    if not etf_list:
        stop_with("Pick at least one hedge candidate.")

    with st.spinner("Downloading prices and running the models..."):
        try:
            tickers = pipeline.tickers_to_download(weights, etf_list)
            returns = get_returns(tuple(tickers), start.isoformat())
            factors = get_factors()
        except Exception as error:  # noqa: BLE001 (any download failure gets the same message)
            stop_with(f"Couldn't download the data. Try again in a minute. ({error})")

        # Tickers Yahoo didn't recognize come back empty. Drop them and say so.
        missing = [t for t in weights.index if returns[t].isna().all()]
        if missing:
            st.warning(f"No price data for {', '.join(missing)}. Check the symbols; "
                       "they were left out.")
            weights = weights.drop(missing)
            if len(weights) < 2:
                stop_with("Fewer than 2 stocks have price data.")
        etf_list = [e for e in etf_list if not returns[e].isna().all()]
        if returns["SPY"].isna().all() or not etf_list:
            stop_with("Couldn't get ETF prices. Try again in a minute.")

        try:
            results = pipeline.run_analysis(weights / weights.sum(), returns, factors,
                                            notional, etf_list)
        except ValueError as error:
            stop_with(str(error))

        st.session_state["results"] = results
        st.session_state["figures"] = pipeline.make_charts(results)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if "results" not in st.session_state:
    st.info("Results show up here after you run the analysis.")
    st.stop()

r = st.session_state["results"]
figures = st.session_state["figures"]
st.divider()

one, two, three, four = st.columns(4)
one.metric("Market beta", f"{r['market_beta']:.2f}",
           help="How much the book moves when the market moves 1%.")
two.metric("Actual volatility", f"{r['realized_vol']:.0%}", help="Annualized.")
three.metric("Best hedge", f"{r['best_hedge']} × {r['hedge_ratio']:.2f}",
             help=f"Short ${r['short_dollars']:,.0f} against a ${r['notional']:,.0f} book.")
four.metric("Independent bets", f"{r['pca']['effective_bets']:.1f} of {r['n_stocks']}",
            help="How many truly different bets the book holds. All moving as one = 1.")

shares = r["factor_shares"]
top = shares.drop("Specific").idxmax()
if shares[top] > 0.5:
    st.warning(f"Concentrated: {shares[top]:.0%} of your risk comes from one factor ({top}).")
else:
    st.success(f"No single factor drives more than half your risk. Largest: {top} "
               f"({shares[top]:.0%}).")

returns_tab, risk_tab, hedge_tab, math_tab = st.tabs(
    ["Returns", "Risk", "Hedge", "Why factor models"])

with returns_tab:
    st.pyplot(figures["returns"])
    st.markdown("**Where the yearly excess return came from**")
    attribution = r["attribution"].to_frame("contribution per year")
    st.dataframe(as_percent(attribution, ["contribution per year"], 1))
    st.markdown("**Factor betas and fit by stock (Fama-French + momentum)**")
    st.dataframe(pd.concat([r["stock_betas"], r["stock_stats"]], axis=1).round(2))

with risk_tab:
    st.pyplot(figures["risk"])
    st.markdown("**The three models, graded against what actually happened**")
    st.dataframe(as_percent(r["comparison"], ["predicted vol", "actual vol"], 1))
    st.markdown("**Money vs risk by position**")
    st.dataframe(as_percent(r["positions"], ["weight", "risk share"], 1))

with hedge_tab:
    st.pyplot(figures["hedge"])
    st.write(f"Shorting **${r['short_dollars']:,.0f} of {r['best_hedge']}** moves the book's "
             f"beta to SPY from **{r['beta_before']:.2f}** to **{r['beta_after']:.2f}**, and "
             f"volatility from **{r['realized_vol']:.0%}** to **{r['vol_after']:.0%}**.")
    st.dataframe(as_percent(r["ranking"], ["R²", "vol before", "vol after"]).round(2))

with math_tab:
    st.markdown(
        "A risk model needs a covariance matrix: how every stock moves with every other "
        "stock. Estimating it straight from price history breaks down fast. It needs "
        "$N(N+1)/2$ numbers, and it can't even be inverted unless there are more days of "
        "history than stocks. Factor models estimate a few betas per stock instead.")
    n = r["n_stocks"]
    yours = risk.count_numbers_needed(n, 6)
    universe = risk.count_numbers_needed(6000, 6)
    st.table(pd.DataFrame(
        {"Full covariance matrix": [f"{yours[0]:,}", f"{universe[0]:,}"],
         "6-factor model": [f"{yours[1]:,}", f"{universe[1]:,}"]},
        index=[f"Your book ({n} stocks)", "6,000-stock universe"]))
    st.markdown("Every model here builds the covariance matrix the same way:")
    st.latex(r"\Sigma = B F B^\top + D")
    st.markdown("And it's guaranteed to be positive definite. For any portfolio $x \\neq 0$:")
    st.latex(r"x^\top \Sigma x = \underbrace{(B^\top x)^\top F (B^\top x)}_{\ge 0}"
             r" + \underbrace{\textstyle\sum_i d_i x_i^2}_{> 0} > 0")
    st.markdown("So no portfolio can show zero or negative risk, every eigenvalue is "
                "positive, and the matrix can always be inverted. Full proofs are in "
                "`MATH.md` in the GitHub repo.")

st.caption("Prices from Yahoo Finance. Factors from the Kenneth French Data Library, which "
           "runs a month or two behind. Estimates are in-sample. Not investment advice.")
