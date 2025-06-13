import json, numpy as np, pandas as pd, streamlit as st, yfinance as yf
from datetime import date, timedelta
from price_chart_cfg import seriesLineChart, COLOR_BULL, chartMultipaneOptions
from streamlit_lightweight_charts import renderLightweightCharts

from portfolio_analysis import calc_portfolio_metrics, get_latest_news
from stock_list import load_portfolio_data

#  1.  Tab entry-point
def portfolio_page() -> None:

    upl = st.file_uploader(
        "Upload a CSV with **Ticker, BuyDate, Shares** "
        "(try *test_portfolio.csv* to start)", type="csv")
    if upl is None:
        st.stop()

    # upl = "test_portfolio.csv"

    pf = pd.read_csv(upl, parse_dates=["BuyDate"])

    st.dataframe(pf)

    values, prices = load_portfolio_data(pf)

    # st.dataframe(prices)
    # st.dataframe(values)


    entry_value = 0.0  # aggregated cost basis via yfinance close
    for _, row in pf.iterrows():
        tkr, buy_dt, sh = row["Ticker"], row["BuyDate"], row["Shares"]

        # find the first available close on/after BuyDate
        entry_price = prices.loc[prices.index >= buy_dt, tkr].iloc[0]
        entry_value += entry_price * sh

        values[tkr] = values.get(tkr, 0) + prices[tkr] * sh

    values["Total"] = values.sum(axis=1)

    # st.dataframe(values)

    #  4.  Metrics
    current_val = values["Total"].iloc[-1]
    delta_abs   = current_val - entry_value
    delta_pct   = delta_abs / entry_value

    c1, c2, c3 = st.columns(3)
    c1.metric("Cost basis",   f"${entry_value:,.2f}")
    c2.metric("Current value",f"${current_val:,.2f}",
              delta=f"${delta_abs:,.2f}",
              delta_color = "normal" if delta_abs > 0 else "inverse")
    c3.metric("Return",       f"{delta_pct*100:,.2f} %")

    #  5.  Single-pane line chart
    tmp = (
        values["Total"]  # Series indexed by daily dates
        .rename_axis("time")  # give the index a name
        .reset_index(name="value")  # → DataFrame with 'time' + 'value'
    )

    # convert the datetime column to the YYYY-MM-DD strings the chart wants
    tmp["time"] = tmp["time"].dt.strftime("%Y-%m-%d")

    portfolio_json = json.loads(tmp.to_json(orient="records"))

    renderLightweightCharts([{
        "chart": chartMultipaneOptions[0],
        "series": seriesLineChart(portfolio_json, line_color=COLOR_BULL)
    }], key="portfolio_chart")

    # --- метрики портфеля --------------------------------------------------

    st.header("Portfolio metrics")

    metrics_df = calc_portfolio_metrics(values["Total"])
    orient = {
        "CAGR": True,
        "Sharpe": True,
        "Sortino": True,
        "Volatility": False,
        "Max DD": False,
    }

    # простые пороги «хорошо/плохо» (можно потом подстроить)
    good_thresh = {
        "CAGR": 0.05,  # ≥ 5 % годовых
        "Sharpe": 1.0,
        "Sortino": 1.0,
        "Volatility": 0.15,  # ≤ 15 %
        "Max DD": -0.10,  # просадка ≥ –10 % (то есть меньше по модулю)
    }

    # ───────────── dashboards cards without st.metric ──────────────

    order = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD"]

    def pretty_val(name: str, val: float) -> str:
        """Format number with % if needed."""
        if name in {"CAGR", "Volatility", "Max DD"}:
            return f"{val:+.2%}"
        return f"{val:.2f}"

    def card(col, name: str, val: float) -> None:
        """Render one coloured card in the given column."""
        better_high = orient[name]
        good = (val >= good_thresh[name]) if better_high else (val <= good_thresh[name])
        tag = "green" if good else "red"

        col.markdown(f"#### {name}")  # title
        col.markdown(f":{tag}[**{pretty_val(name, val)}**]")

    # iterate metrics two per row
    it = iter(order)
    for m1 in it:
        col1, col2 = st.columns(2, gap="large")

        # left card
        v1 = metrics_df.loc[metrics_df.Metric.eq(m1), "Value"].iat[0]
        card(col1, m1, v1)

        # right card (if any left)
        try:
            m2 = next(it)
            v2 = metrics_df.loc[metrics_df.Metric.eq(m2), "Value"].iat[0]
            card(col2, m2, v2)
        except StopIteration:
            pass

    # ----- per-ticker position table ---------------------------------
    st.subheader("Positions")

    rows = []
    for tkr in pf["Ticker"].unique():
        # total shares held
        shares = pf.loc[pf.Ticker == tkr, "Shares"].sum()

        # cost basis for this ticker
        spent = 0.0
        for _, r in pf.loc[pf.Ticker == tkr].iterrows():
            buy_px = prices.loc[prices.index >= r["BuyDate"], tkr].iloc[0]
            spent += buy_px * r["Shares"]

        # latest close
        last_px = prices[tkr].iloc[-1]

        # current market value
        cur_val = last_px * shares

        # profit / loss
        diff = cur_val - spent
        diff_p = diff / spent if spent else 0.0
        tag = "green" if diff >= 0 else "red"

        rows.append({
            "Ticker": tkr,
            "Shares": f"{shares:,.2f}",
            "Last Px": f"${last_px:,.2f}",
            "Cost": f"${spent:,.2f}",
            "Value": f"${cur_val:,.2f}",
            "P/L": f":{tag}[${diff:,.2f}]",
            "P/L %": f":{tag}[{diff_p * 100:,.2f}%]"
        })

    pos_df = pd.DataFrame(rows)

    # render with markdown so colour tags show
    def md_row(r):
        return "| " + " | ".join(str(v) for v in r) + " |"

    header = "| Ticker | Shares | Last Px | Cost | Value | P/L | P/L % |\n"
    header += "|---|---|---|---|---|---|---|"

    st.markdown(header + "\n" + "\n".join(md_row(r) for r in pos_df.values))


    # news
    # news_dict = get_latest_news(pf["Ticker"].tolist(), per_ticker=3)
    #
    # st.subheader("Latest headlines")
    # for tkr, items in news_dict.items():
    #     st.markdown(f"**{tkr}**")
    #     if not items:
    #         st.write("_No news available_")
    #         continue
    #
    #     for art in items:
    #         # ------------------ дата (если есть) ------------------
    #         ts_raw = art.get("providerPublishTime") or art.get("provider_publish_time")
    #         prefix = ""
    #         if ts_raw:
    #             ts = pd.to_datetime(ts_raw, unit="s").strftime("%Y-%m-%d")
    #             prefix = f"**{ts}** – "
    #
    #         # ------------------ заголовок и ссылка -----------------
    #         title = art.get("title") or art.get("headline") or "Untitled"
    #         link = art.get("link") or art.get("url") or "#"
    #
    #         st.markdown(f"- {prefix}[{title}]({link})")

