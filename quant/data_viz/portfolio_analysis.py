import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252               # year norm

def calc_portfolio_metrics(
    curve: pd.Series,
    risk_free_rate: float = 0.02
) -> pd.DataFrame:

    curve = curve.sort_index()
    ret = curve.pct_change().dropna()

    # вспомогательные величины
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    excess_ret = ret - risk_free_rate / TRADING_DAYS

    cagr = (curve.iloc[-1] / curve.iloc[0])**(1 / years) - 1
    vol  = ret.std(ddof=0) * np.sqrt(TRADING_DAYS)
    sharpe = (excess_ret.mean() /
              excess_ret.std(ddof=0)) * np.sqrt(TRADING_DAYS)

    downside = ret[ret < 0]
    sortino = (ret.mean() /
               downside.std(ddof=0)) * np.sqrt(TRADING_DAYS) if len(downside) else np.nan

    roll_max = curve.cummax()
    max_dd = (curve / roll_max - 1).min()

    return pd.DataFrame(
        {"Metric": ["CAGR", "Volatility", "Sharpe",
                    "Sortino", "Max DD"],
         "Value":  [cagr, vol, sharpe, sortino, max_dd]}
    )

def get_latest_news(tickers: list[str], per_ticker: int = 5) -> dict:
    """
    Возвращает {ticker: list(dict(title, link, publisher, providerPublishTime))}
    """
    news = {}
    for t in tickers:
        try:
            news[t] = yf.Ticker(t).news[:per_ticker]
        except Exception:
            news[t] = []
    return news
