"""
charts.py: the three charts that tell the story.
    1. Where are our returns coming from?
    2. Where is our risk coming from, and are we concentrated?
    3. Which ETF tracks our book the best?
"""
import matplotlib

matplotlib.use("Agg")  # save straight to files, no pop-up windows
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, GREY, RED, GREEN, BLACK = (
    "#2563eb", "#f59e0b", "#9ca3af", "#dc2626", "#16a34a", "#111827")

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titleweight": "bold",
    "font.size": 10,
})


def as_percent(ax, axis="y"):
    # Show 0.25 as 25% on the axis
    def to_percent(x, _):
        return f"{x:.0%}"

    if axis == "y":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(to_percent))
    else:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(to_percent))


def clean_dates(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def signed(x):
    return "0%" if abs(x) < 0.005 else f"{x:+.0%}"


def finish(fig, path):
    # Save to a file for the terminal version, or hand the figure back for the website
    if path:
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig


def add_titles(fig, title, subtitle):
    fig.suptitle(title, fontsize=17, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.955, subtitle, color="#4b5563", fontsize=10.5)


# ---------------------------------------------------------------------------
# 1. Returns
# ---------------------------------------------------------------------------

def returns_chart(model_r2, attribution, stock_betas, subtitle, path=None):
    fig, ax = plt.subplot_mosaic([["models", "waterfall"], ["betas", "betas"]],
                                 figsize=(15, 11), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.94))

    # (a) How much of the daily moves each model explains
    a = ax["models"]
    bars = a.bar(model_r2.index, model_r2.values, color=[GREY, BLUE, ORANGE], width=0.55)
    for bar, value in zip(bars, model_r2.values):
        a.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.0%}",
               ha="center", va="bottom", fontweight="bold")
    a.set_ylim(0, 1)
    as_percent(a)
    a.set_title("How much of the book's daily moves each model explains (R²)")
    a.grid(axis="x", visible=False)

    # (b) Waterfall: each factor's piece of the annual return, then the total
    a = ax["waterfall"]
    running = 0
    for i, (name, value) in enumerate(attribution.items()):
        a.bar(i, value, bottom=running, color=GREEN if value >= 0 else RED, width=0.6)
        a.text(i, max(running, running + value), signed(value), ha="center", va="bottom",
               fontsize=9)
        running += value
    a.bar(len(attribution), running, color=BLACK, width=0.6)
    a.text(len(attribution), max(running, 0), signed(running), ha="center", va="bottom",
           fontweight="bold")
    a.axhline(0, color="black", lw=0.8)
    a.set_xticks(range(len(attribution) + 1), list(attribution.index) + ["Total"],
                 rotation=30, ha="right")
    as_percent(a)
    a.set_title("Where our yearly excess return came from (Fama-French + momentum)")
    a.grid(axis="x", visible=False)

    # (c) Heatmap of each stock's factor betas
    a = ax["betas"]
    betas = stock_betas.T  # factors as rows, stocks as columns
    colors = betas.copy()
    colors.loc["Market"] = colors.loc["Market"] - 1  # color market beta vs 1, not vs 0
    image = a.imshow(colors.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    a.set_xticks(range(betas.shape[1]), betas.columns)
    a.set_yticks(range(betas.shape[0]), betas.index)
    for i in range(betas.shape[0]):
        for j in range(betas.shape[1]):
            a.text(j, i, f"{betas.values[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                   color="white" if abs(colors.values[i, j]) > 0.6 else "black")
    a.grid(False)
    a.set_title("Factor betas by stock (red = leaning into the factor, blue = against it; "
                "market colored vs β = 1)")
    fig.colorbar(image, ax=a, shrink=0.85)

    add_titles(fig, "1. Where are our returns coming from?", subtitle)
    return finish(fig, path)


# ---------------------------------------------------------------------------
# 2. Risk
# ---------------------------------------------------------------------------

def risk_chart(factor_shares, pca, positions, model_vols, realized, subtitle, path=None):
    fig, ax = plt.subplot_mosaic([["factors", "pca"], ["positions", "models"]],
                                 figsize=(15, 11), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.94))

    # (a) Share of risk from each factor, flag it if one factor runs the show
    a = ax["factors"]
    shares = factor_shares.sort_values()
    top = factor_shares.drop("Specific").idxmax()
    concentrated = factor_shares[top] > 0.5
    colors = [RED if (name == top and concentrated) else ORANGE if name == "Specific"
              else BLUE for name in shares.index]
    a.barh(shares.index, shares.values, color=colors)
    for i, value in enumerate(shares.values):
        a.text(max(value, 0), i, "  " + signed(value).lstrip("+"), va="center")
    a.axvline(0.5, color=GREY, ls="--", lw=1)
    a.set_xlim(min(0, shares.min() * 1.2), 1)
    as_percent(a, "x")
    flag = (f"CONCENTRATED: {factor_shares[top]:.0%} of our risk is {top}" if concentrated
            else f"Biggest factor: {top} ({factor_shares[top]:.0%})")
    a.set_title(f"Share of our risk by factor\n{flag}", color=RED if concentrated else BLACK)
    a.grid(axis="y", visible=False)

    # (b) PCA: how many independent bets do we really have?
    a = ax["pca"]
    explained = pca["explained"]
    is_real = explained * pca["n_stocks"] > pca["noise_edge"]
    a.bar(range(1, len(explained) + 1), explained.values,
          color=np.where(is_real, BLUE, GREY))
    a.axhline(pca["noise_edge"] / pca["n_stocks"], color=RED, ls="--", lw=1.2,
              label="Noise line (Marchenko-Pastur): below it is just noise")
    a.set_xticks(range(1, len(explained) + 1))
    a.set_xlabel("Statistical factor (principal component)")
    as_percent(a)
    a.set_title(f"How many independent bets? {pca['effective_bets']:.1f} out of "
                f"{pca['n_stocks']} names\nPC1 alone explains {explained.iloc[0]:.0%}")
    a.legend(frameon=False, fontsize=9)
    a.grid(axis="x", visible=False)

    # (c) Capital vs risk by position
    a = ax["positions"]
    p = positions.sort_values("risk share")
    y = np.arange(len(p))
    heavy = p["risk share"] > 1.15 * p["weight"]
    a.barh(y - 0.2, p["weight"], height=0.4, color=GREY, label="% of our money")
    a.barh(y + 0.2, p["risk share"], height=0.4, color=BLUE, label="% of our risk")
    if heavy.any():  # paint the troublemakers red
        a.barh(y[heavy] + 0.2, p["risk share"][heavy], height=0.4, color=RED,
               label="Way more risk than money (>1.15x)")
    a.set_yticks(y, p.index)
    as_percent(a, "x")
    a.set_title("Money vs risk by position")
    a.legend(frameon=False, fontsize=9, loc="lower right")
    a.grid(axis="y", visible=False)

    # (d) Grading the three models against what actually happened
    a = ax["models"]
    bars = a.bar(model_vols.index, model_vols.values, color=[GREY, BLUE, ORANGE], width=0.55)
    for bar, value in zip(bars, model_vols.values):
        a.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0%}", ha="center",
               va="bottom", fontweight="bold")
    a.axhline(realized, color=RED, lw=2, label=f"Actual volatility: {realized:.0%}")
    a.set_ylim(0, max(model_vols.max(), realized) * 1.2)
    as_percent(a)
    a.set_title("Predicted volatility by model vs what actually happened")
    a.legend(frameon=False, loc="upper left")
    a.grid(axis="x", visible=False)

    add_titles(fig, "2. Where is our risk coming from, and are we concentrated?", subtitle)
    return finish(fig, path)


# ---------------------------------------------------------------------------
# 3. Hedge
# ---------------------------------------------------------------------------

def growth(returns):
    return (1 + returns).cumprod()


def hedge_chart(portfolio, etf_returns, ranking, notional, subtitle, path=None):
    fig, ax = plt.subplot_mosaic([["main", "main"], ["monthly", "ranking"],
                                  ["correlation", "hedged"]],
                                 figsize=(15, 15), height_ratios=[1.3, 1, 1],
                                 layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.95))
    best = ranking.index[0]
    h = ranking["hedge ratio"]

    # (a) The headline: our book vs each ETF scaled by its hedge ratio.
    # We add a constant so each line has the same average return as the book.
    # That way the chart shows who TRACKS us, not who happened to go up more.
    def replica(etf):
        drift = portfolio.mean() - h[etf] * etf_returns[etf].mean()
        return growth(h[etf] * etf_returns[etf] + drift)

    a = ax["main"]
    a.plot(growth(portfolio), color=BLACK, lw=2.8, label="Our book", zorder=5)
    a.plot(replica(best), color=BLUE, lw=2.2, zorder=4,
           label=f"{best} × {h[best]:.2f}  (best match: R² {ranking.loc[best, 'R²']:.0%})")
    others = [etf for etf in ["SMH", "QQQ", "SPY"] if etf in ranking.index and etf != best]
    for etf, style, color in zip(others, ["--", ":", "-."], [ORANGE, "#8b5cf6", GREY]):
        a.plot(replica(etf), color=color, lw=1.4, ls=style,
               label=f"{etf} × {h[etf]:.2f}  (R² {ranking.loc[etf, 'R²']:.0%})")
    a.axhline(1, color="black", lw=0.6)
    a.set_ylabel("Growth of $1")
    a.set_title("Our book vs each ETF scaled by its hedge ratio "
                "(the line that hugs ours is the best hedge)")
    a.legend(frameon=False, loc="upper left")
    clean_dates(a)

    # (b) Rolling 1-month returns (21 trading days). No drift, just "do we move together?"
    a = ax["monthly"]
    a.plot(portfolio.rolling(21).sum(), color=BLACK, lw=1.8, label="Our book")
    a.plot((h[best] * etf_returns[best]).rolling(21).sum(), color=BLUE, lw=1.6,
           label=f"{best} × {h[best]:.2f}")
    a.axhline(0, color="black", lw=0.6)
    as_percent(a)
    a.set_title("Rolling 1-month returns: our book vs the best hedge")
    a.legend(frameon=False)
    clean_dates(a)

    # (c) Leaderboard: how much of our daily moves each ETF matches
    a = ax["ranking"]
    r = ranking.sort_values("R²")
    a.barh(r.index, r["R²"], color=[BLUE if etf == best else GREY for etf in r.index])
    for i, (etf, row) in enumerate(r.iterrows()):
        a.text(row["R²"], i, f"  {row['R²']:.0%}   short {row['hedge ratio']:.2f}x",
               va="center", fontsize=9)
    a.set_xlim(0, 1)
    as_percent(a, "x")
    a.set_title("Tracking leaderboard (R² = correlation²)")
    a.grid(axis="y", visible=False)

    # (d) Does the tracking hold up over time?
    a = ax["correlation"]
    for etf, color in zip(ranking.index[:3], [BLUE, ORANGE, "#8b5cf6"]):
        a.plot(portfolio.rolling(63).corr(etf_returns[etf]), color=color, lw=1.6, label=etf)
    a.set_ylim(0, 1)
    a.set_title("Rolling 3-month correlation with our book (top 3)")
    a.legend(frameon=False, ncol=3)
    clean_dates(a)

    # (e) What's left after the hedge
    a = ax["hedged"]
    hedged = portfolio - h[best] * etf_returns[best]
    vol_before = portfolio.std() * np.sqrt(252)
    vol_after = hedged.std() * np.sqrt(252)
    a.plot(growth(portfolio), color=GREY, lw=1.5, label="Unhedged")
    a.plot(growth(hedged), color=GREEN, lw=2,
           label=f"Hedged: short ${h[best] * notional:,.0f} of {best}")
    a.axhline(1, color="black", lw=0.6)
    a.set_title(f"What's left after the hedge: volatility {vol_before:.0%} → {vol_after:.0%}")
    a.legend(frameon=False)
    clean_dates(a)

    add_titles(fig, "3. Which ETF tracks our book the best?", subtitle)
    return finish(fig, path)
