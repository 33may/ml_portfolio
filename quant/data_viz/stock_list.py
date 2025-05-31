from __future__ import annotations
import pandas as pd
from pandas_datareader.stooq import StooqDailyReader
from tqdm.auto import tqdm           # nice progress bar in notebooks and terminal
from dataclasses import dataclass
from typing import Union, Literal, Tuple

# ---------------------------------------------------------------------------
# 0.  Load static S&P-500 table (once) --------------------------------------
# ---------------------------------------------------------------------------

def load_sp500() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0]
    df = df.rename(columns={"Symbol": "Ticker", "Security": "Name"})
    return df[["Ticker", "Name", "GICS Sector", "GICS Sub-Industry"]]

df_sp_500: pd.DataFrame = load_sp500()

# Convenience helpers -------------------------------------------------------

def all_tickers()       -> list[str]: return df_sp_500["Ticker"].tolist()
def all_sectors()       -> list[str]: return df_sp_500["GICS Sector"].unique().tolist()
def all_sub_industry()  -> list[str]: return df_sp_500["GICS Sub-Industry"].unique().tolist()

# ---------------------------------------------------------------------------
# 1.  One-shot price loader with tqdm ---------------------------------------
# ---------------------------------------------------------------------------

def _fetch_price_and_diff(ticker: str) -> Tuple[float | None, float | None]:
    """
    Return (last_close, diff_30days) for a single ticker via Stooq.

    diff_30days is (last_close - close_30days_ago) / last_close
    """
    try:
        df = StooqDailyReader(symbols=ticker,
                              retry_count=3,
                              pause=0.1).read()
        if df.empty or len(df) < 31:             # insure we have ≥31 rows
            return None, None

        last_close      = float(df["Close"].iloc[-1])
        prev_30_close   = float(df["Close"].iloc[-30])
        diff            = (last_close - prev_30_close) / last_close
        return last_close, diff
    except Exception:
        return None, None

def preload_prices(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Add Price and Dif columns to df_base in-place.  A tqdm bar is shown.
    """
    df_base[["Price", "Dif"]] = None        # pre-create blank cols

    for idx, ticker in tqdm(enumerate(df_base["Ticker"]),
                            total=len(df_base),
                            desc="Fetching Stooq prices"):
        price, diff = _fetch_price_and_diff(ticker)
        df_base.at[idx, "Price"] = price
        df_base.at[idx, "Dif"]   = diff

    return df_base

# Do the heavy work once at module import time ------------------------------
# (takes ~15-20 s over a typical connection).
df_sp_500 = preload_prices(df_sp_500)

# ---------------------------------------------------------------------------
# 2.  Query object and paginated accessor -----------------------------------
# ---------------------------------------------------------------------------

@dataclass
class Query:
    name         : Union[str, None]                 # substring in Name
    page         : int                              # 1-based page
    sort         : Literal["price_top", "price_bottom"]
    sector       : Union[str, None]                 # exact GICS Sector match or None
    sub_industry : Union[str, None]                 # exact Sub-Industry match or None

def get_stocks_paginated(query: Query,
                         page_size: int = 10) -> pd.DataFrame:
    """
    Filter cached S&P-500 table by the Query, then return the requested page
    (already containing Price and Dif columns downloaded earlier).

    Page indexing is **1-based**:
        page=1  -> rows 0 … page_size-1
        page=2  -> rows page_size … 2*page_size-1
    """
    mask = pd.Series(True, index=df_sp_500.index)

    if query.sector:
        mask &= df_sp_500["GICS Sector"] == query.sector

    if query.sub_industry:
        mask &= df_sp_500["GICS Sub-Industry"] == query.sub_industry

    if query.name:
        mask &= df_sp_500["Name"].str.contains(query.name, case=False, na=False)

    df_filtered = df_sp_500[mask].copy()

    # --- sorting -----------------------------------------------------------
    ascending = query.sort == "price_bottom"
    df_filtered = df_filtered.sort_values(by="Price",
                                          ascending=ascending,
                                          na_position="last")

    # --- pagination --------------------------------------------------------
    start = (query.page - 1) * page_size
    end   = start + page_size
    return df_filtered.iloc[start:end].reset_index(drop=True)
