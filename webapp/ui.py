"""
ui.py: the look of the website. Colors, the CSS on top of Streamlit's theme, and the
small building blocks every page uses (page header, cards, callouts, number formats).

The palette is white and light grey, with one soft red that marks risk and the thing
worth looking at. Everything else stays grey so the red means something.
"""
import html

import streamlit as st

RED = "#D9534F"          # the accent: risk, the best pick, the thing to look at
RED_TEXT = "#B83C38"     # red that's dark enough to read as text
RED_SOFT = "#FCEDEC"     # red wash for backgrounds
INK = "#38424D"          # your book, totals
SLATE = "#4F7396"        # the cool opposite of red (positive contributions, low betas)
GREY = "#A3ABB5"         # context
GREY_LIGHT = "#D5DAE0"
TEXT = "#1F2328"
TEXT_2 = "#5B6470"
MUTED = "#8A929C"
GRID = "#EEF0F3"
AXIS = "#D5DAE0"
BORDER = "#E6E8EC"
PAGE_GREY = "#F6F7F9"

CSS = f"""
<style>
/* ---- Sidebar: tabs on the left, like a dashboard ---- */
[data-testid="stSidebarHeader"] {{ padding: 1.6rem 1.4rem 1rem 1.4rem; }}
[data-testid="stSidebarNav"] {{ padding: 0 0.75rem; }}
[data-testid="stSidebarNavLink"] {{
    padding: 0.55rem 0.85rem; margin: 2px 0; border-radius: 10px;
}}
[data-testid="stSidebarNavLink"] span {{ font-size: 0.97rem; color: {TEXT}; }}
[data-testid="stSidebarNavLink"]:hover {{ background: {PAGE_GREY}; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: {RED_SOFT}; }}
[data-testid="stSidebarNavLink"][aria-current="page"] span {{
    color: {RED_TEXT}; font-weight: 600;
}}
[data-testid="stNavSectionHeader"] {{
    font-size: 0.7rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: {MUTED}; font-weight: 600; padding: 1.1rem 0.85rem 0.3rem 0.85rem;
}}
[data-testid="stSidebarNavSeparator"] {{ display: none; }}

/* ---- Main area ---- */
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{
    max-width: 1280px; padding-top: 2.2rem; padding-bottom: 4rem;
}}

@media (max-width: 640px) {{
    /* On phones the sidebar collapses and the logo sits on top: leave room for it */
    [data-testid="stMainBlockContainer"] {{ padding-top: 4.5rem; }}
}}

/* Page header: title + subtitle on the left, a context tag on the right */
.page-head {{
    display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem;
    flex-wrap: wrap; padding-bottom: 1.1rem; margin-bottom: 0.4rem;
    border-bottom: 1px solid {BORDER};
}}
.page-title {{ font-size: 1.85rem; font-weight: 800; color: {TEXT}; line-height: 1.15; }}
.page-sub {{ font-size: 0.95rem; color: {TEXT_2}; margin-top: 0.3rem; }}
.page-tag {{
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    color: {TEXT_2}; border: 1px solid {BORDER}; border-radius: 999px;
    padding: 0.45rem 0.9rem; white-space: nowrap;
}}

/* Cards: any container whose key starts with "card" */
[class*="st-key-card"] {{
    background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 14px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 4px 14px rgba(16, 24, 40, 0.04);
    padding: 1.25rem 1.35rem 1.1rem 1.35rem;
}}
.eyebrow {{
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
    color: {MUTED};
}}
.card-title {{ font-size: 1.12rem; font-weight: 700; color: {TEXT}; margin-top: 0.2rem; }}
.card-note {{ font-size: 0.9rem; color: {TEXT_2}; margin-top: 0.25rem; line-height: 1.5; }}
.section {{
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
    color: {MUTED}; margin: 1.6rem 0 0.2rem 0;
}}

/* Callouts: red when something needs attention, grey when it's just information */
[class*="st-key-callout"] {{
    border-radius: 12px; padding: 0.9rem 1.1rem; border: 1px solid {BORDER};
    background: {PAGE_GREY};
}}
[class*="st-key-callout_risk"] {{
    background: {RED_SOFT}; border-color: #F5D0CE; border-left: 4px solid {RED};
}}
[class*="st-key-callout"] p {{ margin-bottom: 0; }}
[class*="st-key-callout"] [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}

/* Multiselect tags: soft red pills, not solid blocks */
[data-testid="stMultiSelectTagsContainer"] > span > span {{
    background: {RED_SOFT}; color: {RED_TEXT}; font-weight: 600;
}}
[data-testid="stMultiSelectTagsContainer"] > span > span * {{ color: {RED_TEXT}; }}

/* KPI tiles */
[data-testid="stMetric"] {{
    background: #FFFFFF; border-radius: 14px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}}
[data-testid="stMetricLabel"] p {{
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    color: {MUTED};
}}

/* Small stat lines inside model cards */
.stat-row {{
    display: flex; justify-content: space-between; gap: 0.75rem; padding: 0.55rem 0;
    border-top: 1px solid {GRID}; font-size: 0.9rem; color: {TEXT_2};
}}
.stat-row b {{ color: {TEXT}; font-weight: 600; white-space: nowrap; }}
.stat-row.win b {{ color: {RED_TEXT}; }}
</style>
"""


def apply_style():
    st.html(CSS)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def page_header(title, subtitle, tag=None):
    tag_html = f'<div class="page-tag">{html.escape(tag)}</div>' if tag else ""
    st.html(f'<div class="page-head"><div><div class="page-title">{html.escape(title)}</div>'
            f'<div class="page-sub">{subtitle}</div></div>{tag_html}</div>')


def section(label):
    st.html(f'<div class="section">{html.escape(label)}</div>')


def card(key):
    # A white card. The key has to start with "card" for the CSS to find it.
    return st.container(key=f"card_{key}")


def card_header(eyebrow, title, note=None):
    note_html = f'<div class="card-note">{note}</div>' if note else ""
    st.html(f'<div class="eyebrow">{html.escape(eyebrow)}</div>'
            f'<div class="card-title">{html.escape(title)}</div>{note_html}')


def callout(key, text, risk=False):
    # Red for "look at this", grey for "good to know"
    with st.container(key=f"callout_{'risk' if risk else 'info'}_{key}"):
        st.markdown(text)


def kpi(column, label, value, note, help=None):
    # A KPI tile: label, big number, and a short grey note underneath
    column.metric(label, value, delta=note, delta_color="off", delta_arrow="off",
                  border=True, help=help)


def stat_rows(rows):
    # rows: (label, value, highlighted) tuples, shown as a tidy two-column list
    st.html("".join(
        f'<div class="stat-row{" win" if win else ""}"><span>{html.escape(label)}</span>'
        f'<b>{html.escape(value)}</b></div>' for label, value, win in rows))


# ---------------------------------------------------------------------------
# Number formats
# ---------------------------------------------------------------------------

def pct(x, decimals=0):
    return f"{x:.{decimals}%}"


def signed_pct(x, decimals=1):
    # +12.3% / −4.0%, with a real minus sign
    text = f"{x:+.{decimals}%}"
    return text.replace("-", "−")


def money(x):
    return f"${x:,.0f}"
