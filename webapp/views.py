"""
views.py: the pages of the website, one function per tab in the sidebar.

    Portfolio      enter holdings + settings, run everything
    Overview       headline numbers and which model explains the book best
    CAPM / Fama-French / Statistical
                   one page per model, same layout so they're easy to compare
    Hedge          which ETF tracks the book, how much to short
    How it works   why factor models, the math in brief

All the math comes from riskmodel/pipeline.py, same as the terminal version.
"""
from datetime import date

import pandas as pd
import streamlit as st

from riskmodel import data, hedge, pipeline, risk

from . import charts, ui
from .charts import show

MAX_STOCKS = 50
BLANK_ROWS = 8

# A made-up book so first-time visitors can see the site work in one click
EXAMPLE = pd.DataFrame({
    "ticker": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM"],
    "weight": [15.0, 15.0, 15.0, 12.0, 12.0, 11.0, 10.0, 10.0],
})

MODELS = {
    "CAPM": {
        "title": "CAPM",
        "subtitle": "One factor: the stock market. How much of your book is just the market?",
        "card": "One factor, the market. The simplest yardstick.",
        "page": "capm",
        "alpha": "the part the market can't explain: stock picking, or risks CAPM "
                 "doesn't measure.",
    },
    "Fama-French": {
        "title": "Fama-French + momentum",
        "subtitle": "Six factors: market, size, value, profitability, investment and momentum.",
        "card": "Six academic factors, including value, size and momentum.",
        "page": "fama_french",
        "alpha": "the part none of the six factors explain: stock picking, or a theme the "
                 "model doesn't have (like a sector).",
    },
    "Statistical": {
        "title": "Statistical (PCA)",
        "subtitle": "No preset factors. The data finds what your stocks have in common.",
        "card": "Factors found in your own stocks' returns.",
        "page": "statistical",
        "alpha": "the part the shared moves don't explain: each stock's own story.",
    },
}

# Which side of each Fama-French factor a positive beta leans toward
LEANS = {
    "Size": ("small caps", "large caps"),
    "Value": ("cheap (value) stocks", "expensive (growth) stocks"),
    "Profitability": ("profitable companies", "unprofitable companies"),
    "Investment": ("conservative spenders", "aggressive spenders"),
    "Momentum": ("recent winners", "recent losers"),
}


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def pages():
    # st.Page needs a running script, so these get built on every run
    return {
        "portfolio": st.Page(portfolio, title="Portfolio", icon=":material/work_outline:",
                             url_path="portfolio", default=True),
        "overview": st.Page(overview, title="Overview", icon=":material/space_dashboard:",
                            url_path="overview"),
        "capm": st.Page(capm, title="CAPM", icon=":material/show_chart:", url_path="capm"),
        "fama_french": st.Page(fama_french, title="Fama-French",
                               icon=":material/stacked_bar_chart:", url_path="fama-french"),
        "statistical": st.Page(statistical, title="Statistical (PCA)",
                               icon=":material/scatter_plot:", url_path="statistical"),
        "hedge": st.Page(hedge_page, title="Hedge", icon=":material/shield:",
                         url_path="hedge"),
        "method": st.Page(how_it_works, title="How it works", icon=":material/menu_book:",
                          url_path="how-it-works"),
    }


def sidebar():
    p = pages()
    return {
        "": [p["portfolio"], p["overview"]],
        "Models": [p["capm"], p["fama_french"], p["statistical"]],
        "Hedging": [p["hedge"]],
        "Learn": [p["method"]],
    }


def header(title, subtitle):
    # Page title, plus a tag on the right saying which book the numbers are for
    r = st.session_state.get("results")
    tag = None
    if r is not None:
        start, end = r["factor_dates"]
        tag = f"{r['n_stocks']} holdings · {start:%b %Y} – {end:%b %Y}"
    ui.page_header(title, subtitle, tag)


def require_results():
    r = st.session_state.get("results")
    if r is None:
        with ui.card("empty"):
            ui.card_header("Nothing to show yet", "Start with your portfolio",
                           "Enter your holdings and run the analysis. Every tab fills in "
                           "from there.")
            st.page_link(pages()["portfolio"], label="Enter your holdings",
                         icon=":material/arrow_forward:")
        st.stop()
    return r


# ---------------------------------------------------------------------------
# Data (downloads are slow, so we remember them: prices for an hour, factors for a day)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_returns(tickers, start_date):
    return data.to_returns(data.load_prices(list(tickers), start_date))


@st.cache_data(ttl=86400, show_spinner=False)
def get_factors():
    return data.load_factors()


def blank_holdings():
    return pd.DataFrame({"ticker": pd.Series([None] * BLANK_ROWS, dtype="object"),
                         "weight": pd.Series([None] * BLANK_ROWS, dtype="float")})


def clean_holdings(table):
    # Tidy up whatever got typed in: uppercase tickers, skip blank rows, merge duplicates
    table = table.dropna(subset=["ticker", "weight"]).copy()
    table["ticker"] = table["ticker"].astype(str).str.strip().str.upper()
    table = table[table["ticker"] != ""]
    return table.groupby("ticker")["weight"].sum()


def reset_holdings(table):
    # A new editor key throws away the old table's edits and starts from this one
    st.session_state.holdings_saved = table.copy()
    st.session_state.holdings_version += 1


def stop_with(message):
    st.error(message, icon=":material/error:")
    st.stop()


# ---------------------------------------------------------------------------
# Portfolio: the input page
# ---------------------------------------------------------------------------

def portfolio():
    ui.page_header("Portfolio", "Enter your holdings, then run all three factor models on "
                                "them.")
    st.session_state.setdefault("holdings_saved", blank_holdings())
    st.session_state.setdefault("holdings_version", 0)

    if "results" in st.session_state:
        ui.callout("last_run",
                   f":material/check_circle: Results are ready for your "
                   f"{st.session_state.results['n_stocks']} holdings. Change anything below "
                   "and run again, or open the **Overview** in the sidebar.")

    left, right = st.columns([3, 2], gap="large")
    with left, ui.card("holdings"):
        ui.card_header("Step 1", "Your holdings",
                       "One row per stock. Weights can be percentages, decimals or dollar "
                       "amounts; they get scaled to add up to 100%.")
        # The table forgets its edits when you switch tabs, so we keep a copy and start
        # from that copy whenever the table is drawn fresh.
        key = f"holdings_{st.session_state.holdings_version}"
        if key not in st.session_state:
            st.session_state.holdings_base = st.session_state.holdings_saved
        edited = st.data_editor(
            st.session_state.holdings_base, key=key, num_rows="dynamic", hide_index=True,
            width="stretch", placeholder="",
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", help="e.g. AAPL"),
                "weight": st.column_config.NumberColumn("Weight", min_value=0.0,
                                                        format="%.2f"),
            })
        st.session_state.holdings_saved = edited.reset_index(drop=True)
        with st.container(horizontal=True):
            st.button("Load an example", icon=":material/auto_awesome:",
                      on_click=reset_holdings, args=(EXAMPLE,))
            st.button("Clear", icon=":material/delete_sweep:", on_click=reset_holdings,
                      args=(blank_holdings(),))
        st.caption(f"Long positions only. 2 to {MAX_STOCKS} stocks.")

    with right, ui.card("settings"):
        ui.card_header("Step 2", "Settings")
        start = st.date_input("History starts", value=date(2023, 1, 1),
                              min_value=date(2005, 1, 1), max_value=date.today(),
                              key="start", persist_state="session")
        notional = st.number_input("Book size ($)", min_value=1_000, value=100_000,
                                   step=1_000, key="notional", persist_state="session",
                                   help="Used to turn the hedge ratio into dollars to short.")
        etfs = st.multiselect("Hedge candidates", options=hedge.CANDIDATE_ETFS,
                              default=hedge.CANDIDATE_ETFS, key="etfs",
                              persist_state="session")
        extra = st.text_input("Other ETFs to test (optional)", placeholder="e.g. ARKK, XLE",
                              key="extra", persist_state="session")
        run = st.button("Run analysis", type="primary", width="stretch",
                        icon=":material/play_arrow:")

    if run:
        analyze(st.session_state.holdings_saved, start, notional, etfs, extra)
        st.switch_page(pages()["overview"])


def analyze(holdings, start, notional, etfs, extra):
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

    notes = []
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
            notes.append(f"No price data for {', '.join(missing)}. Check the symbols; "
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

    st.session_state.results = results
    st.session_state.notes = notes


# ---------------------------------------------------------------------------
# Overview: headline numbers + which model wins
# ---------------------------------------------------------------------------

def overview():
    header("Overview", "How your book behaves, and which model explains it best.")
    r = require_results()
    for i, note in enumerate(st.session_state.get("notes", [])):
        ui.callout(f"note_{i}", f":material/info: {note}", risk=True)

    one, two, three, four = st.columns(4)
    ui.kpi(one, "Market beta", f"{r['market_beta']:.2f}",
           f"{r['market_beta']:.2f}% per 1% market move",
           help="How much the book moves when the market moves 1% (CAPM).")
    ui.kpi(two, "Actual volatility", ui.pct(r["realized_vol"], 1), "per year",
           help="How much the book actually swung, annualized. Every model's forecast is "
                "graded against this.")
    ui.kpi(three, "Independent bets", f"{r['pca']['effective_bets']:.1f}",
           f"out of {r['n_stocks']} stocks",
           help="How many truly different bets the book holds. All moving as one = 1.")
    ui.kpi(four, "Best hedge", r["best_hedge"],
           f"short {ui.money(r['short_dollars'])} ({r['hedge_ratio']:.2f}×)",
           help="The ETF that tracks the book most closely. See the Hedge tab.")

    shares = r["factor_shares"]
    top = shares.drop("Specific").idxmax()
    if shares[top] > 0.5:
        ui.callout("concentration",
                   f":material/warning: **Concentrated.** {ui.pct(shares[top])} of your risk "
                   f"comes from one factor ({top}), per the Fama-French model.", risk=True)
    else:
        ui.callout("concentration",
                   f":material/check_circle: No single factor drives more than half your "
                   f"risk. The largest is {top} at {ui.pct(shares[top])} (Fama-French).")

    ui.section("Which model explains your book best?")
    comparison, winners, best = r["comparison"], r["winners"], r["best_model"]
    points = pd.Series(winners).value_counts()
    if points[best] > 1:
        verdict = (f"**{MODELS[best]['title']} wins {points[best]} of 3 tests**: "
                   + ", ".join(test.lower() for test, name in winners.items() if name == best)
                   + ".")
    else:
        verdict = (f"**Close call.** Each model wins one test, so the tie goes to the one that "
                   f"explains the most of your daily moves: **{MODELS[best]['title']}**.")
    if best == "Statistical":
        verdict += (" Keep in mind its factors are built from your own stocks, so it fits "
                    "them almost by definition. CAPM and Fama-French use outside factors, "
                    "which is what lets them say *what kind* of risk you hold.")
    ui.callout("verdict", f":material/emoji_events: {verdict}")

    columns = st.columns(3)
    for column, name in zip(columns, MODELS):
        m = r["by_model"][name]
        with column, ui.card(f"model_{MODELS[name]['page']}"):
            wins = points.get(name, 0)
            if name == best:
                st.badge(f"Best overall · wins {wins} of 3", color="red",
                         icon=":material/emoji_events:")
            else:
                st.badge(f"Wins {wins} of 3", color="gray")
            ui.card_header("Model", MODELS[name]["title"], MODELS[name]["card"])
            ui.stat_rows([
                ("Explains daily moves", ui.pct(m["r2"]),
                 winners["Explains the most"] == name),
                ("Predicted volatility", ui.pct(m["predicted_vol"], 1),
                 winners["Closest risk forecast"] == name),
                ("Leftover correlation", f"{m['leftover_correlation']:.2f}",
                 winners["Cleanest leftovers"] == name),
                ("Alpha left unexplained", ui.signed_pct(m["attribution"]["Alpha"]) + "/yr",
                 False),
                ("Factors", str(m["n_factors"]), False),
            ])
            st.page_link(pages()[MODELS[name]["page"]], label=f"Open {name}",
                         icon=":material/arrow_forward:")

    left, right = st.columns(2, gap="medium")
    with left, ui.card("r2"):
        ui.card_header("Fit", "Share of your daily moves each model explains",
                       "R². Higher means the model's factors capture more of what your book "
                       "does day to day.")
        show(*charts.highlight_bars(comparison["R²"], winners["Explains the most"], "model",
                                    "R²", domain=[0, 1]))
    with right, ui.card("vol"):
        ui.card_header("Risk forecast", "Predicted volatility vs what actually happened",
                       "Each model's Σ = BFBᵀ + D turned into a volatility forecast. Closest "
                       "to the line wins.")
        top_vol = max(comparison["predicted vol"].max(), r["realized_vol"]) * 1.25
        show(*charts.highlight_bars(comparison["predicted vol"],
                                    winners["Closest risk forecast"], "model",
                                    "Predicted volatility", domain=[0, top_vol],
                                    rule=r["realized_vol"],
                                    rule_label=f"Actual {ui.pct(r['realized_vol'], 1)}",
                                    text=[ui.pct(v, 1) for v in comparison["predicted vol"]]))

    with ui.card("explained"):
        total = r["by_model"]["CAPM"]["attribution"].sum()
        ui.card_header("Returns", "How each model splits your yearly return",
                       f"Your book beat T-bills by {ui.signed_pct(total)}/yr. Each model "
                       "splits that into what its factors explain and what's left over "
                       "(alpha). Less alpha = the model explains more of your return.")
        split = pd.DataFrame({
            "Explained by factors": {n: m["attribution"].drop("Alpha").sum()
                                     for n, m in r["by_model"].items()},
            "Alpha (unexplained)": {n: m["attribution"]["Alpha"]
                                    for n, m in r["by_model"].items()},
        })
        show(*charts.grouped_bars(split, [ui.INK, ui.GREY_LIGHT], "Per year"))

    with st.expander("Scorecard table"):
        st.dataframe(comparison, column_config={
            "R²": st.column_config.NumberColumn(format="percent"),
            "predicted vol": st.column_config.NumberColumn(format="percent"),
            "actual vol": st.column_config.NumberColumn(format="percent"),
            "leftover correlation": st.column_config.NumberColumn(format="%.2f"),
        })


# ---------------------------------------------------------------------------
# The three model pages (same layout, so they're easy to compare)
# ---------------------------------------------------------------------------

def capm():
    model_page("CAPM")


def fama_french():
    model_page("Fama-French")


def statistical():
    model_page("Statistical")


def model_page(name):
    info = MODELS[name]
    header(info["title"], info["subtitle"])
    r = require_results()
    m = r["by_model"][name]
    actual = r["realized_vol"]
    alpha = m["attribution"]["Alpha"]

    if r["best_model"] == name:
        st.badge("Best overall for your book", color="red", icon=":material/emoji_events:")

    one, two, three, four = st.columns(4)
    ui.kpi(one, "Explains", ui.pct(m["r2"]), "of your daily moves (R²)",
           help="Share of your book's daily moves this model's factors capture.")
    ui.kpi(two, "Predicted volatility", ui.pct(m["predicted_vol"], 1),
           f"{(m['predicted_vol'] - actual) * 100:+.1f} pts vs actual {ui.pct(actual, 1)}"
           .replace("-", "−"),
           help="The model's risk forecast for your book, annualized.")
    ui.kpi(three, "Alpha", ui.signed_pct(alpha), "per year, unexplained",
           help="Yearly return this model can't explain with its factors.")
    ui.kpi(four, "Factors", str(m["n_factors"]),
           f"{m['numbers_estimated']:,} numbers estimated",
           help=f"A full covariance matrix for {r['n_stocks']} stocks needs "
                f"{risk.count_numbers_needed(r['n_stocks'], 1)[0]:,} numbers.")

    # ---- Returns ----
    ui.section("Returns")
    left, right = st.columns([3, 2], gap="medium")
    with left, ui.card(f"returns_{info['page']}"):
        ui.card_header("Return attribution", "Where your yearly return came from",
                       "Exposure × what each factor earned, plus alpha. The pieces add up "
                       "exactly to the total.")
        show(*charts.contribution_bars(m["attribution"]))
    with right, ui.card(f"explain_{info['page']}"):
        ui.card_header("In plain English", "What drove your return")
        st.markdown(explain_returns(name, m, r))

    # ---- Risk ----
    ui.section("Risk")
    shares = m["risk_shares"]
    top = shares.drop("Specific").idxmax()
    if shares[top] > 0.5:
        ui.callout(f"conc_{info['page']}",
                   f":material/warning: **Concentrated.** {ui.pct(shares[top])} of your risk "
                   f"comes from one factor ({top}).", risk=True)
    left, right = st.columns(2, gap="medium")
    with left, ui.card(f"risk_{info['page']}"):
        ui.card_header("Risk by factor", "Where your day-to-day swings come from",
                       f"Factors explain {ui.pct(m['r2'])} of your swings. The biggest is "
                       f"{top} at {ui.pct(shares[top])}; the rest is stock-specific.")
        show(*charts.risk_share_bars(shares))
    with right, ui.card(f"positions_{info['page']}"):
        ui.card_header("Risk by position", "Money vs risk in each stock",
                       position_note(m["positions"]))
        show(*charts.money_vs_risk(m["positions"]))

    # ---- Exposures ----
    ui.section("Exposures")
    if name == "CAPM":
        with ui.card("betas_capm"):
            ui.card_header("Market beta", "How hard each stock swings with the market",
                           f"The dark line is β = 1, moving one-for-one with the market. Red "
                           f"bars swing harder. Your whole book's beta is "
                           f"{m['book_betas']['Market']:.2f}.")
            show(*charts.beta_bars(m["stock_betas"]["Market"]))
    elif name == "Fama-French":
        with ui.card("betas_ff"):
            ui.card_header("Factor betas", "Each stock's lean on each factor",
                           "Red = leans into the factor, slate = leans against it. Market "
                           "is shaded against 1 (a normal stock), the rest against 0.")
            show(*charts.heatmap(m["stock_betas"], center_column="Market"))
    else:
        pca = r["pca"]
        left, right = st.columns(2, gap="medium")
        with left, ui.card("scree"):
            ui.card_header("Hidden factors", "How many real factors are in your stocks?",
                           f"{r['n_stat_factors']} above the noise line (Marchenko-Pastur), "
                           f"so the model keeps {r['n_stat_factors']}. Your stocks add up to "
                           f"{pca['effective_bets']:.1f} independent bets out of "
                           f"{pca['n_stocks']}.")
            show(*charts.scree(pca["explained"], pca["noise_edge"] / pca["n_stocks"]))
        with right, ui.card("loadings"):
            names = describe_components(m["stock_betas"])
            ui.card_header("Loadings", "Which stocks drive each factor",
                           " ".join(f"{pc}: {desc}." for pc, desc in names.items()))
            show(*charts.heatmap(m["stock_betas"], value_title="Loading"))

    with st.expander("See the numbers"):
        table = m["stock_betas"].copy()
        if m["stock_stats"] is not None:
            table = pd.concat([table, m["stock_stats"]], axis=1)
        st.markdown("**By stock**")
        st.dataframe(table.round(2))
        st.markdown("**Money vs risk by position**")
        st.dataframe(m["positions"], column_config={
            "weight": st.column_config.NumberColumn(format="percent"),
            "risk share": st.column_config.NumberColumn(format="percent")})


def explain_returns(name, m, r):
    # Turn the attribution into a few sentences anyone can follow
    start, end = r["factor_dates"]
    pieces = m["attribution"].drop("Alpha")
    lines = [f"Your book beat T-bills by **{ui.signed_pct(m['attribution'].sum())}/yr** "
             f"({start:%b %Y} – {end:%b %Y}).", ""]
    names = describe_components(m["stock_betas"]) if name == "Statistical" else {}
    for factor in pieces.abs().sort_values(ascending=False).index:
        value = pieces[factor]
        if abs(value) < 0.005:
            continue  # rounds to nothing, skip it
        beta = m["book_betas"][factor]
        earned = m["factor_returns"][factor]
        if name == "Statistical":
            why = f"{names[factor]}."
        elif factor == "Market":
            why = (f"your beta of {beta:.2f} × the market's {ui.signed_pct(earned)}/yr over "
                   "T-bills.")
        else:
            side = LEANS[factor][0 if beta > 0 else 1]
            winner, loser = LEANS[factor] if earned > 0 else LEANS[factor][::-1]
            why = (f"you lean toward {side} (β {beta:+.2f}), and {winner} beat {loser} by "
                   f"{abs(earned):.1%}/yr.").replace("-", "−")
        lines.append(f"- **{factor} {ui.signed_pct(value)}**: {why}")
    lines.append(f"- **Alpha {ui.signed_pct(m['attribution']['Alpha'])}**: "
                 f"{MODELS[name]['alpha']}")
    return "\n".join(lines)


def describe_components(loadings):
    # Give each principal component a plain-English name from the stocks that drive it
    names = {}
    for pc in loadings.columns:
        column = loadings[pc]
        against = column[column < 0]
        if against.abs().sum() < 0.1 * column.abs().sum():  # (almost) nothing against
            leaders = ", ".join(column.sort_values(ascending=False).index[:3])
            names[pc] = f"your whole book moving together (led by {leaders})"
        else:
            up = " & ".join(column[column > 0].sort_values(ascending=False).index[:2])
            down = " & ".join(against.sort_values().index[:2])
            names[pc] = f"{up} moving against {down}"
    return names


def position_note(positions):
    ratio = positions["risk share"] / positions["weight"]
    worst = ratio.idxmax()
    if ratio[worst] > 1.15:
        return (f"{worst} is {ui.pct(positions.loc[worst, 'weight'])} of your money but "
                f"{ui.pct(positions.loc[worst, 'risk share'])} of your risk.")
    return "No position carries much more risk than its share of your money."


# ---------------------------------------------------------------------------
# Hedge
# ---------------------------------------------------------------------------

def growth(returns):
    return (1 + returns).cumprod()


def hedge_page():
    header("Hedge", "Which ETF tracks your book best, and how much to short to take the "
                    "market out.")
    r = require_results()
    ranking, best = r["ranking"], r["best_hedge"]
    book, etf_returns = r["portfolio"], r["etf_returns"]

    one, two, three, four = st.columns(4)
    ui.kpi(one, "Best match", best, f"tracks {ui.pct(ranking.loc[best, 'R²'])} of daily moves")
    ui.kpi(two, "Short", ui.money(r["short_dollars"]),
           f"{r['hedge_ratio']:.2f}× your {ui.money(r['notional'])} book")
    ui.kpi(three, "Beta to SPY", f"{r['beta_before']:.2f} → {r['beta_after']:.2f}",
           "before → after the hedge", help="Near 0 after the hedge = market neutral.")
    ui.kpi(four, "Volatility", f"{ui.pct(r['realized_vol'])} → {ui.pct(r['vol_after'])}",
           "before → after, per year")

    with ui.card("leaderboard"):
        ui.card_header("Leaderboard", "Which ETF moves most like your book",
                       "R² = correlation². The higher it is, the more of your risk a short "
                       "in that ETF cancels out. Your volatility after hedging is "
                       "volatility × √(1 − R²).")
        text = [f"{row['R²']:.0%}  ·  short {row['hedge ratio']:.2f}×"
                for _, row in ranking.iterrows()]
        show(*charts.highlight_bars(ranking["R²"], best, "ETF", "R²", domain=[0, 1],
                                    text=text))

    ui.section("Compare your book with an ETF")
    etf = st.selectbox("ETF", list(ranking.index), index=0, label_visibility="collapsed",
                       format_func=lambda e: f"{e}  (R² {ranking.loc[e, 'R²']:.0%})")
    h = ranking.loc[etf, "hedge ratio"]

    with ui.card("tracking"):
        ui.card_header("Tracking", f"Your book vs {etf} × {h:.2f}",
                       "The ETF is scaled by its hedge ratio and given the same average "
                       "return as your book, so the chart shows how closely it tracks you, "
                       "not who went up more.")
        drift = book.mean() - h * etf_returns[etf].mean()
        table = pd.DataFrame({"Your book": growth(book),
                              f"{etf} × {h:.2f}": growth(h * etf_returns[etf] + drift)})
        show(*charts.lines(table, [ui.INK, ui.RED], "Growth of $1", zero=1))

    left, right = st.columns(2, gap="medium")
    with left, ui.card("rolling"):
        rolling = book.rolling(63).corr(etf_returns[etf]).dropna()
        ui.card_header("Stability", "Does the match hold up?",
                       f"Correlation with {etf} over rolling 3-month windows. Steady and "
                       "high = a reliable hedge.")
        show(*charts.lines(rolling.to_frame(f"Correlation with {etf}"), [ui.RED],
                           "Correlation", domain=[min(0, rolling.min()), 1]))
    with right, ui.card("hedged"):
        hedged = book - h * etf_returns[etf]
        vol = book.std() * risk.DAYS_PER_YEAR ** 0.5
        vol_after = hedged.std() * risk.DAYS_PER_YEAR ** 0.5
        ui.card_header("After the hedge", "What's left once you short it",
                       f"Shorting {ui.money(h * r['notional'])} of {etf} takes volatility "
                       f"from {ui.pct(vol)} to {ui.pct(vol_after)}. What's left is the part "
                       "of your return the ETF doesn't share.")
        table = pd.DataFrame({"Unhedged": growth(book), f"Hedged with {etf}": growth(hedged)})
        show(*charts.lines(table, [ui.GREY, ui.RED], "Growth of $1", zero=1))

    with st.expander("Full hedge table"):
        st.dataframe(ranking, column_config={
            "correlation": st.column_config.NumberColumn(format="%.2f"),
            "R²": st.column_config.NumberColumn(format="percent"),
            "hedge ratio": st.column_config.NumberColumn(format="%.2f"),
            "vol before": st.column_config.NumberColumn(format="percent"),
            "vol after": st.column_config.NumberColumn(format="percent")})


# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------

def how_it_works():
    header("How it works", "Why factor models, and what each of the three does.")
    r = st.session_state.get("results")

    with ui.card("why"):
        ui.card_header("The problem", "Why not just use the covariance matrix?")
        st.markdown(
            "A risk model needs a covariance matrix: how every stock moves with every other "
            "stock. Estimating it straight from price history breaks down fast. It needs "
            "$N(N+1)/2$ numbers, and it can't even be inverted unless there are more days "
            "of history than stocks. Factor models estimate a few betas per stock instead.")
        n = r["n_stocks"] if r else 12
        yours = risk.count_numbers_needed(n, 6)
        universe = risk.count_numbers_needed(6000, 6)
        st.table(pd.DataFrame(
            {"Full covariance matrix": [f"{yours[0]:,}", f"{universe[0]:,}"],
             "6-factor model": [f"{yours[1]:,}", f"{universe[1]:,}"]},
            index=[f"{'Your' if r else 'A'} book ({n} stocks)", "6,000-stock universe"]))

    with ui.card("sigma"):
        ui.card_header("The fix", "Every model builds risk the same way")
        st.latex(r"\Sigma = B F B^\top + D")
        st.markdown("$B$ = each stock's factor betas, $F$ = how the factors move together, "
                    "$D$ = each stock's own (specific) risk. It's guaranteed to be positive "
                    "definite. For any portfolio $x \\neq 0$:")
        st.latex(r"x^\top \Sigma x = \underbrace{(B^\top x)^\top F (B^\top x)}_{\ge 0}"
                 r" + \underbrace{\textstyle\sum_i d_i x_i^2}_{> 0} > 0")
        st.markdown("So no portfolio can show zero or negative risk, and the matrix can "
                    "always be inverted. Full proofs are in `MATH.md` in the GitHub repo.")

    columns = st.columns(3)
    blurbs = {
        "CAPM": "Regress each stock on the market. One beta per stock: how hard it swings "
                "when the market moves.",
        "Fama-French": "Same regression with six factors: market, size, value, "
                       "profitability, investment and momentum. Says *what kind* of "
                       "stocks you own.",
        "Statistical": "No preset factors. Eigenvectors of your stocks' covariance, keeping "
                       "only those above the Marchenko-Pastur noise line "
                       "$(1 + \\sqrt{N/T})^2$.",
    }
    for column, (name, text) in zip(columns, blurbs.items()):
        with column, ui.card(f"about_{MODELS[name]['page']}"):
            ui.card_header("Model", MODELS[name]["title"])
            st.markdown(text)

    with ui.card("grading"):
        ui.card_header("Grading", "How the best model is picked")
        st.markdown(
            "Three tests, one point each. Most points wins; a tie goes to the higher R².\n"
            "- **Explains the most**: highest R², the share of your book's daily moves "
            "its factors capture.\n"
            "- **Closest risk forecast**: predicted volatility nearest to what actually "
            "happened.\n"
            "- **Cleanest leftovers**: factor models assume what's left over is unrelated "
            "across stocks. We average the size of the correlation between every pair of "
            "leftovers; the closer to 0, the less the model missed.")

    st.caption("Prices from Yahoo Finance. Factors from the Kenneth French Data Library, "
               "which runs a month or two behind. Estimates are in-sample. Not investment "
               "advice.")
