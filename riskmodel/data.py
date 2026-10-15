"""
data.py: getting the raw stuff.

Prices come from Yahoo Finance. Factor returns come from Ken French's data
library (free, and it's the same data the academic papers use).
"""
import io
import zipfile

import pandas as pd
import requests
import yfinance as yf

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def load_portfolio(path="portfolio.csv"):
    # Our book: one row per ticker with its weight. Scale so the weights add to 100%.
    table = pd.read_csv(path)
    weights = table.set_index("ticker")["weight"]
    return weights / weights.sum()


def load_prices(tickers, start_date):
    # Daily closes, adjusted for splits and dividends so the returns are real returns.
    prices = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)["Close"]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    # reindex = a ticker Yahoo doesn't recognize shows up as an empty column instead of crashing
    return prices.reindex(columns=tickers)


def to_returns(prices):
    # Price -> daily % return. Stocks that listed late (SNDK, ALAB) are just blank before day one.
    return prices.pct_change(fill_method=None).iloc[1:]


def parse_french_text(text):
    # French's CSVs have a paragraph of text on top and a copyright line at the bottom.
    # We only keep the rows that start with a date like 20240102.
    lines = text.splitlines()
    header = None
    rows = []
    for i, line in enumerate(lines):
        cells = [cell.strip() for cell in line.split(",")]
        is_date_row = len(cells[0]) == 8 and cells[0].isdigit()
        if is_date_row:
            if header is None:
                # The column names sit right above the first date
                header = [cell.strip() for cell in lines[i - 1].split(",")]
            rows.append(cells)
        elif rows:
            break  # we were in the table and it just ended

    table = pd.DataFrame(rows, columns=header)
    table.index = pd.to_datetime(table.iloc[:, 0], format="%Y%m%d")
    table.index.name = "date"
    return table.iloc[:, 1:].astype(float) / 100  # French reports in %, we want decimals


def read_french_file(file_name):
    response = requests.get(FRENCH_URL + file_name, timeout=30)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zipped:
        text = zipped.read(zipped.namelist()[0]).decode("latin-1")
    return parse_french_text(text)


def load_factors():
    # Daily Fama-French 5 factors + momentum. Quick cheat sheet:
    #   Mkt-RF = the market minus T-bills
    #   SMB    = small caps minus big caps
    #   HML    = value minus growth
    #   RMW    = profitable companies minus unprofitable ones
    #   CMA    = conservative spenders minus aggressive spenders
    #   UMD    = recent winners minus recent losers (momentum)
    #   RF     = the risk-free rate (T-bills)
    five_factors = read_french_file("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
    momentum = read_french_file("F-F_Momentum_Factor_daily_CSV.zip")
    momentum.columns = ["UMD"]
    return five_factors.join(momentum, how="inner")
