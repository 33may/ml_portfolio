from __future__ import annotations

import time

from stockdex import Ticker

from datetime import date, timedelta, datetime

import pandas as pd
from pandas_datareader.stooq import StooqDailyReader
from tqdm.auto import tqdm           # nice progress bar in notebooks and terminal
from dataclasses import dataclass
from typing import Union, Literal, Tuple, Optional, Any


# ---------------------------------------------------------------------------
# 0.  Load static S&P-500 table (once) --------------------------------------
# ---------------------------------------------------------------------------

def load_sp500() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0]
    df = df.rename(columns={"Symbol": "Ticker", "Security": "Name"})
    df = df[df["Name"] != "BRK.B"]

    test = True

    if test:
        df = df[:10]


    return df[["Ticker", "Name", "GICS Sector", "GICS Sub-Industry"]]

df_sp_500: pd.DataFrame = load_sp500()

# Convenience helpers -------------------------------------------------------

def all_tickers()       -> list[str]: return df_sp_500["Ticker"].tolist()
def all_sectors()       -> list[str]: return df_sp_500["GICS Sector"].unique().tolist()
def all_sub_industry()  -> list[str]: return df_sp_500["GICS Sub-Industry"].unique().tolist()

# ---------------------------------------------------------------------------
# 1.  One-shot price loader with tqdm ---------------------------------------
# ---------------------------------------------------------------------------

def _fetch_price_and_diff(ticker_symbol: str, retries: int = 3, backoff_factor: float = 1.0) -> None | tuple[
    None, None] | tuple[Any, Any]:
    """
    Fetches the latest closing price and calculates the percentage change over the past 30 days.
    Implements a retry mechanism with exponential backoff for handling ReadTimeout errors.

    :param ticker_symbol: Stock ticker symbol.
    :param retries: Number of retry attempts.
    :param backoff_factor: Factor for exponential backoff.
    :return: Tuple containing the latest closing price and percentage change.
    """
    for attempt in range(1, retries + 1):
        try:
            ticker = Ticker(ticker=ticker_symbol)
            price_df = ticker.yahoo_api_price(range='1mo', dataGranularity='1d')
            if price_df.empty or len(price_df) < 2:
                return None, None
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
            price_df = price_df.sort_values('timestamp')
            first_close = price_df.iloc[0]['close']
            last_close = price_df.iloc[-1]['close']
            diff = (last_close - first_close) / last_close
            return last_close, diff
        except Exception as e:
            print(f"Attempt {attempt} failed for {ticker_symbol}: {e}")
            if attempt == retries:
                return None, None
            sleep_time = backoff_factor * (2 ** (attempt - 1))
            print(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)


def preload_prices(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Add Price and Dif columns to df_base in-place.  A tqdm bar is shown.
    """
    df_base[["Price", "Dif"]] = None

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
    sort         : Union[Literal["price_top", "price_bottom"], None]
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
