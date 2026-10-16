"""
charts.py: the website's charts, one question per chart.

Built with Altair so they're interactive (hover any bar or line for the exact number)
and stretch to fit the page. The terminal version still uses riskmodel/charts.py.
"""
import altair as alt
import pandas as pd
import streamlit as st

from .ui import AXIS, GREY, GREY_LIGHT, GRID, INK, MUTED, RED, SLATE, TEXT, TEXT_2

FONT = "Inter, system-ui, -apple-system, sans-serif"
BAR = 22  # bar thickness in px: thin marks, the rest is air
PERCENT = ".0%"


def show(chart, height):
    chart = (chart.properties(height=height, padding={"left": 0, "right": 12, "top": 6,
                                                       "bottom": 0})
             .configure(font=FONT, background="transparent")
             .configure_view(stroke=None)
             .configure_axis(labelColor=TEXT_2, titleColor=MUTED, gridColor=GRID,
                             domainColor=AXIS, tickColor=AXIS, labelFontSize=12,
                             titleFontSize=11, titleFontWeight=500, labelPadding=6,
                             tickSize=4, labelLimit=180)
             .configure_legend(orient="top", title=None, labelColor=TEXT_2, labelFontSize=12,
                               symbolType="square", symbolSize=110, columnPadding=18,
                               padding=0, offset=10, labelLimit=320)
             .configure_text(font=FONT, color=TEXT_2, fontSize=12))
    st.altair_chart(chart, width="stretch", theme=None)


def rows(series, label, value="value"):
    return series.rename(value).rename_axis(label).reset_index()


def field_name(name):
    # Vega reads "." and "[" in a column name as "look inside", so escape them
    return name.replace(".", "\\.").replace("[", "\\[").replace("]", "\\]")


def tip_label(x):
    return f"{x:+.1%}".replace("-", "−")


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------

def highlight_bars(values, highlight, label, value_title, value_format=PERCENT,
                   domain=None, rule=None, rule_label=None, height=None, text=None):
    # One bar per row, the highlighted one in red and the rest grey. Optional
    # reference line (e.g. "actual volatility"). Value printed at the tip of each bar.
    df = rows(values, label)
    df["highlight"] = df[label].isin([highlight] if isinstance(highlight, str) else highlight)
    df["text"] = text if text is not None else df["value"].map(lambda v: format(v, value_format))
    x = alt.X("value:Q", title=value_title, axis=alt.Axis(format=value_format, tickCount=5),
              scale=alt.Scale(domain=domain) if domain else alt.Undefined)
    y = alt.Y(f"{label}:N", sort=None, title=None, axis=alt.Axis(ticks=False, domain=False))
    base = alt.Chart(df).encode(y=y, tooltip=[alt.Tooltip(f"{label}:N"),
                                              alt.Tooltip("value:Q", title=value_title,
                                                          format=value_format)])
    bars = base.mark_bar(size=BAR, cornerRadiusEnd=4).encode(
        x=x, color=alt.condition("datum.highlight", alt.value(RED), alt.value(GREY_LIGHT)))
    labels = base.mark_text(align="left", dx=6, fontWeight=600).encode(
        x="value:Q", text="text:N",
        color=alt.condition("datum.highlight", alt.value(TEXT), alt.value(TEXT_2)))
    layers = [bars, labels]
    if rule is not None:
        line = alt.Chart(pd.DataFrame({"x": [rule], "t": [rule_label]}))
        layers.append(line.mark_rule(color=INK, strokeWidth=2).encode(x="x:Q"))
        layers.append(line.mark_text(align="left", dx=6, dy=-6, color=INK, fontWeight=600,
                                     baseline="bottom")
                      .encode(x="x:Q", y=alt.value(0), text="t:N"))
    return alt.layer(*layers), height or max(150, 46 * len(df) + 40)


def contribution_bars(contributions, total_label="Total"):
    # Where the yearly return came from: one bar per piece (slate = added, red = took
    # away), then the total in dark ink. The pieces add up exactly to the total.
    df = rows(contributions, "piece")
    df = pd.concat([df, pd.DataFrame({"piece": [total_label],
                                      "value": [contributions.sum()]})], ignore_index=True)
    df["kind"] = ["Total" if p == total_label else "Added" if v >= 0 else "Took away"
                  for p, v in zip(df["piece"], df["value"])]
    df["text"] = df["value"].map(tip_label)
    df["label_x"] = df["value"].clip(lower=0)  # labels sit right of the bar, or of 0
    y = alt.Y("piece:N", sort=None, title=None, axis=alt.Axis(ticks=False, domain=False))
    base = alt.Chart(df).encode(
        y=y, tooltip=[alt.Tooltip("piece:N", title="Piece"),
                      alt.Tooltip("value:Q", title="Per year", format="+.1%")])
    bars = base.mark_bar(size=BAR, cornerRadiusEnd=4).encode(
        x=alt.X("value:Q", title="Contribution per year", axis=alt.Axis(format=PERCENT)),
        color=alt.Color("kind:N", legend=None, scale=alt.Scale(
            domain=["Added", "Took away", "Total"], range=[SLATE, RED, INK])))
    labels = base.mark_text(align="left", dx=6, fontWeight=600, color=TEXT).encode(
        x="label_x:Q", text="text:N")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=AXIS).encode(x="x:Q")
    return alt.layer(bars, zero, labels), 46 * len(df) + 40


def risk_share_bars(shares):
    # Share of risk by factor. The biggest factor is red (that's where the risk is),
    # the stock-specific piece is light grey, the rest are grey.
    shares = shares.sort_values(ascending=False)
    top = shares.drop("Specific").idxmax()
    df = rows(shares, "factor")
    df["kind"] = ["Biggest factor" if f == top else "Stock-specific" if f == "Specific"
                  else "Other factors" for f in df["factor"]]
    df["text"] = df["value"].map(lambda v: f"{v:.0%}".replace("-0%", "0%").replace("-", "−"))
    df["label_x"] = df["value"].clip(lower=0)
    kinds = [k for k in ["Biggest factor", "Other factors", "Stock-specific"]
             if k in set(df["kind"])]
    colors = {"Biggest factor": RED, "Other factors": GREY, "Stock-specific": GREY_LIGHT}
    y = alt.Y("factor:N", sort=None, title=None, axis=alt.Axis(ticks=False, domain=False))
    base = alt.Chart(df).encode(
        y=y, tooltip=[alt.Tooltip("factor:N", title="Source"),
                      alt.Tooltip("value:Q", title="Share of risk", format=".1%")])
    bars = base.mark_bar(size=BAR, cornerRadiusEnd=4).encode(
        x=alt.X("value:Q", title="Share of the book's day-to-day swings",
                axis=alt.Axis(format=PERCENT),
                scale=alt.Scale(domain=[min(0, df["value"].min()), max(1, df["value"].max())])),
        color=alt.Color("kind:N", scale=alt.Scale(
            domain=kinds, range=[colors[k] for k in kinds])))
    labels = base.mark_text(align="left", dx=6, fontWeight=600, color=TEXT).encode(
        x="label_x:Q", text="text:N")
    half = alt.Chart(pd.DataFrame({"x": [0.5], "t": ["50%: one factor runs the show"]}))
    rule = half.mark_rule(color=MUTED, strokeDash=[4, 4]).encode(x="x:Q")
    rule_text = half.mark_text(align="left", dx=6, baseline="top", color=MUTED,
                               fontSize=11).encode(x="x:Q", y=alt.value(0), text="t:N")
    return alt.layer(bars, rule, rule_text, labels), 44 * len(df) + 60


def money_vs_risk(positions):
    # Two bars per stock: share of the money (grey) vs share of the risk (red).
    # A red bar much longer than its grey one = a position punching above its size.
    p = positions.sort_values("risk share", ascending=False)
    order = list(p.index)
    df = (p.rename(columns={"weight": "Money", "risk share": "Risk"})
          .rename_axis("stock").reset_index()
          .melt(id_vars="stock", var_name="measure", value_name="value"))
    ratio = (p["risk share"] / p["weight"]).rename("ratio")
    df = df.merge(ratio.rename_axis("stock").reset_index(), on="stock")
    chart = alt.Chart(df).mark_bar(size=11, cornerRadiusEnd=3).encode(
        y=alt.Y("stock:N", sort=order, title=None, axis=alt.Axis(ticks=False, domain=False)),
        yOffset=alt.YOffset("measure:N", sort=["Money", "Risk"]),
        x=alt.X("value:Q", title=None, axis=alt.Axis(format=PERCENT)),
        color=alt.Color("measure:N", sort=["Money", "Risk"], scale=alt.Scale(
            domain=["Money", "Risk"], range=[GREY_LIGHT, RED]),
            legend=alt.Legend(labelExpr="datum.label == 'Money' ? '% of your money' : "
                                        "'% of your risk'")),
        tooltip=[alt.Tooltip("stock:N", title="Stock"),
                 alt.Tooltip("measure:N", title="Share of"),
                 alt.Tooltip("value:Q", title="Share", format=".1%"),
                 alt.Tooltip("ratio:Q", title="Risk ÷ money", format=".2f")])
    return chart, 30 * len(order) + 70


def beta_bars(betas):
    # CAPM betas by stock. Above 1 = swings harder than the market (red).
    s = betas.sort_values(ascending=False)
    df = rows(s, "stock")
    df["kind"] = ["Swings more than the market" if b > 1 else "Swings less than the market"
                  for b in df["value"]]
    df["text"] = df["value"].map(lambda b: f"{b:.2f}")
    base = alt.Chart(df).encode(
        y=alt.Y("stock:N", sort=None, title=None, axis=alt.Axis(ticks=False, domain=False)),
        tooltip=[alt.Tooltip("stock:N", title="Stock"),
                 alt.Tooltip("value:Q", title="Market beta", format=".2f")])
    bars = base.mark_bar(size=16, cornerRadiusEnd=4).encode(
        x=alt.X("value:Q", title="Market beta (β)", axis=alt.Axis(tickCount=6)),
        color=alt.Color("kind:N", scale=alt.Scale(
            domain=["Swings more than the market", "Swings less than the market"],
            range=[RED, GREY_LIGHT])))
    labels = base.mark_text(align="left", dx=5, fontWeight=600, color=TEXT).encode(
        x="value:Q", text="text:N")
    rule = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(color=INK, strokeWidth=1.5).encode(
        x="x:Q")
    return alt.layer(bars, rule, labels), 30 * len(df) + 70


def heatmap(table, center_column=None, value_title="Beta"):
    # Rows = stocks, columns = factors. Red = leans into the factor, slate = leans
    # against it, grey = neutral. center_column gets colored against 1 instead of 0
    # (a market beta of 1 is "normal").
    df = table.rename_axis("stock").reset_index().melt(
        id_vars="stock", var_name="factor", value_name="value")
    df["shade"] = df["value"] - (df["factor"] == center_column).astype(float)
    df["text"] = df["value"].map(lambda v: f"{v:.2f}")
    limit = max(0.5, float(df["shade"].abs().quantile(0.95)))
    base = alt.Chart(df).encode(
        x=alt.X("factor:N", sort=list(table.columns), title=None,
                axis=alt.Axis(orient="top", ticks=False, domain=False, labelAngle=0,
                              labelFontWeight=600)),
        y=alt.Y("stock:N", sort=list(table.index), title=None,
                axis=alt.Axis(ticks=False, domain=False)))
    cells = base.mark_rect(cornerRadius=4, stroke="white", strokeWidth=2).encode(
        color=alt.Color("shade:Q", legend=None, scale=alt.Scale(
            domain=[-limit, 0, limit], range=[SLATE, "#F1F2F4", RED], clamp=True,
            interpolate="lab")),
        tooltip=[alt.Tooltip("stock:N", title="Stock"), alt.Tooltip("factor:N", title="Factor"),
                 alt.Tooltip("value:Q", title=value_title, format=".2f")])
    text = base.mark_text(fontSize=12, fontWeight=500).encode(
        text="text:N",
        color=alt.condition(f"abs(datum.shade) > {0.6 * limit}", alt.value("white"),
                            alt.value(TEXT)))
    return alt.layer(cells, text), 34 * len(table.index) + 40


def scree(explained, noise_share):
    # How much of the stocks' movement each principal component explains. Above the
    # noise line = real structure (red), below it = indistinguishable from noise.
    df = rows(explained, "component")
    df["kind"] = ["Real factor" if v > noise_share else "Noise" for v in df["value"]]
    df["n"] = range(1, len(df) + 1)
    base = alt.Chart(df).encode(
        x=alt.X("n:O", title="Principal component", axis=alt.Axis(labelAngle=0, ticks=False)),
        tooltip=[alt.Tooltip("component:N", title="Component"),
                 alt.Tooltip("value:Q", title="Explains", format=".1%"),
                 alt.Tooltip("kind:N", title="Verdict")])
    bars = base.mark_bar(size=BAR, cornerRadiusEnd=4).encode(
        y=alt.Y("value:Q", title="Share of the stocks' movement",
                axis=alt.Axis(format=PERCENT, tickCount=5)),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Real factor", "Noise"],
                                                  range=[RED, GREY_LIGHT])))
    line = alt.Chart(pd.DataFrame({"y": [noise_share], "t": ["Noise line"]}))
    rule = line.mark_rule(color=INK, strokeDash=[4, 4]).encode(y="y:Q")
    rule_text = line.mark_text(align="right", dy=-6, color=INK, fontSize=11, fontWeight=600,
                               baseline="bottom").encode(
        y="y:Q", x=alt.value("width"), text="t:N")
    return alt.layer(bars, rule, rule_text), 300


def grouped_bars(table, colors, value_title):
    # table: rows = groups (models), columns = series. Used for "explained vs alpha".
    series = list(table.columns)
    df = table.rename_axis("group").reset_index().melt(
        id_vars="group", var_name="series", value_name="value")
    chart = alt.Chart(df).mark_bar(size=13, cornerRadiusEnd=3).encode(
        y=alt.Y("group:N", sort=list(table.index), title=None,
                axis=alt.Axis(ticks=False, domain=False)),
        yOffset=alt.YOffset("series:N", sort=series),
        x=alt.X("value:Q", title=value_title, axis=alt.Axis(format=PERCENT)),
        color=alt.Color("series:N", sort=series,
                        scale=alt.Scale(domain=series, range=colors)),
        tooltip=[alt.Tooltip("group:N", title="Model"), alt.Tooltip("series:N", title="Piece"),
                 alt.Tooltip("value:Q", title="Per year", format="+.1%")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=AXIS).encode(x="x:Q")
    return alt.layer(chart, zero), 52 * len(table.index) + 60


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------

def lines(table, colors, value_title, value_format=".2f", zero=None, domain=None):
    # table: index = dates, one column per series. Hover anywhere for a crosshair
    # with every series' value on that day.
    series = list(table.columns)
    df = table.rename_axis("date").reset_index().melt(
        id_vars="date", var_name="series", value_name="value").dropna()
    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover",
                                empty=False, clear="pointerout")
    color = alt.Color("series:N", sort=series, scale=alt.Scale(domain=series, range=colors))
    y_scale = alt.Scale(domain=domain) if domain else alt.Scale(zero=False)
    base = alt.Chart(df).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y", tickCount=6,
                                                    grid=False)),
        y=alt.Y("value:Q", title=value_title, scale=y_scale,
                axis=alt.Axis(format=value_format, tickCount=5)),
        color=color)
    drawn = base.mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round")
    points = base.mark_point(filled=True, size=70, stroke="white", strokeWidth=2).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    wide = table.rename_axis("date").reset_index()
    rule = alt.Chart(wide).mark_rule(color=AXIS).encode(
        x="date:T",
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("date:T", title="Date", format="%b %d, %Y")]
        + [alt.Tooltip(field=field_name(s), type="quantitative", title=s, format=value_format)
           for s in series],
    ).add_params(hover)
    layers = [drawn, points, rule]
    if zero is not None:
        layers.insert(0, alt.Chart(pd.DataFrame({"y": [zero]})).mark_rule(
            color=AXIS, strokeWidth=1).encode(y="y:Q"))
    return alt.layer(*layers), 320
