import json

import numpy as np
import pandas as pd
import streamlit as st
from finta import TA
from streamlit_lightweight_charts import renderLightweightCharts

from price_chart_cfg import seriesMACDchart, \
    chartMultipaneOptions, seriesVolumeChart, seriesCandlestickChart, COLOR_BULL, COLOR_BEAR, seriesLineChart
from stock_list import all_sectors, df_sp_500, all_sub_industry, Query, get_stocks_paginated

def load_test_data():
    return pd.read_csv("data_yf.csv")


# tiny helper for coloured arrow
def _fmt_diff(diff):
    if diff is None:
        return "–"
    arrow, colour = ("▲", "green") if diff > 0 else ("▼", "red")
    return f":{colour}[{arrow} {abs(diff)*100:,.2f}%]"

def stock_list() -> None:
    st.title("S&P 500 mini-screener")

    # Filters (inside main page)
    with st.expander("Filters", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            # sector
            sector_sel = st.selectbox(
                "Sector",
                ["All"] + all_sectors(),
                key="sector"
            )
        with c2:
            # sub-industries depend on sector selection
            if sector_sel != "All":
                sub_opts = (["All"] +
                            sorted(df_sp_500.loc[
                                   df_sp_500["GICS Sector"] == sector_sel,
                                   "GICS Sub-Industry"].unique()))
            else:
                sub_opts = ["All"] + all_sub_industry()
            sub_sel = st.selectbox("Sub-Industry", sub_opts, key="sub")

        name_val = st.text_input("Search company name", key="name").strip() or None

        sort_val = st.radio(
            "Sort by price",
            ["price_top", "price_bottom"],
            horizontal=True,
            format_func=lambda v: "Top ↓" if v == "price_top" else "Bottom ↑",
            key="sort"
        )

    # BUILD Query
    q = Query(
        name=name_val,
        page=st.session_state.get("page", 1),
        sort=sort_val,
        sector=None if sector_sel == "All" else sector_sel,
        sub_industry=None if sub_sel == "All" else sub_sel
    )

    # DATA
    page_df = get_stocks_paginated(q, page_size=10)

    # reset page if filter change produced fewer rows
    if page_df.empty and q.page > 1:
        st.session_state.page = q.page - 1
        st.rerun()

    # TABLE
    import yfinance as yf

    #  CARD LOOP
    if page_df.empty:
        st.info("No matches.")
        st.stop()

    def header_section(r):
        header_cols = st.columns([2, 5, 2, 2])
        with header_cols[0]:
            st.markdown(f"**{r.Ticker}**")
        with header_cols[1]:
            st.markdown(r.Name)
        with header_cols[2]:
            st.markdown(f"${r.Price:,.2f}" if r.Price else "–")
        with header_cols[3]:
            st.markdown(_fmt_diff(r.Dif))

    @st.cache_data
    def load_data(ticker: str, period: str, interval: str):
        return yf.Ticker(ticker).history(period=period, interval = interval)[['Open', 'High', 'Low', 'Close', 'Volume']]

    def graph_details_section(r):
        tab_labels = ["1 Week", "1 Month", "1 Year", "5 Years"]
        tab_args = ["1wk", "1mo", "1y", "5y"]

        # add a dummy first option
        options = ["— select interval —"] + tab_labels
        choice = st.radio(
            label="",
            options=options,
            horizontal=True,
            key=f"period_select_{r.Ticker}"
        )

        if choice != options[0]:
            # map label → yf arg
            idx = tab_labels.index(choice)
            period = tab_args[idx]
            interval = "1d" if period in {"1wk", "1mo"} else "1wk" if period == "1y" else "1mo"

            with st.spinner(f"Loading {choice} data…"):
                df = load_data(r.Ticker, period, interval)

            df = df.reset_index()
            df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df['color'] = np.where(df['open'] > df['close'], COLOR_BEAR, COLOR_BULL)

            # 3) compute MACD & Signal with FinTA
            macd_df = TA.MACD(df, period_fast=6, period_slow=12, signal=5)
            df = pd.concat([df, macd_df], axis=1)

            # 4) derive histogram
            df['HIST'] = df['MACD'] - df['SIGNAL']

            # 5) export JSON series
            candles = json.loads(df.to_json(orient="records"))
            volume = json.loads(
                df.rename(columns={"volume": "value"})
                .to_json(orient="records")
            )
            macd_fast = json.loads(
                df.rename(columns={"MACD": "value"})
                .to_json(orient="records")
            )
            macd_slow = json.loads(
                df.rename(columns={"SIGNAL": "value"})
                .to_json(orient="records")
            )
            macd_hist = json.loads(
                df.rename(columns={"HIST": "value"})
                .to_json(orient="records")
            )

            st.subheader("Multipane Chart with Pandas")

            renderLightweightCharts([
                {
                    "chart": chartMultipaneOptions[0],
                    "series": seriesCandlestickChart(candles)
                },
                {
                    "chart": chartMultipaneOptions[1],
                    "series": seriesVolumeChart(volume)
                },
                {
                    "chart": chartMultipaneOptions[2],
                    "series": seriesMACDchart(macd_fast, macd_slow, macd_hist)
                }
            ], 'multipane')

        else:
            st.markdown("_Select an interval above to load data_")


    for _, r in page_df.iterrows():
        header_section(r)

        # ---- expandable details ---------------------------------------------
        with st.expander("Details & Buy", expanded=False):
            st.markdown("Expander")

            # ---------- price chart with selectable interval ------------------

            graph_details_section(r)

            st.divider()

            # # ---------- BUY form (Quantity & Amount) --------------------------
            # price = float(r.Price or 0)
            # qty_key = f"{r.Ticker}-qty"
            # amt_key = f"{r.Ticker}-amt"
            #
            # # init state once
            # if qty_key not in st.session_state:
            #     st.session_state[qty_key] = 1.0
            #     st.session_state[amt_key] = price
            #
            # def _sync_from_qty():
            #     st.session_state[amt_key] = st.session_state[qty_key] * price
            #
            # def _sync_from_amt():
            #     st.session_state[qty_key] = (
            #         st.session_state[amt_key] / price if price else 0
            #     )
            #
            # col_q, col_a = st.columns(2, gap="medium")
            # with col_q:
            #     st.number_input("Quantity (shares)",
            #                     min_value=0.0, step=0.1,
            #                     key=qty_key, on_change=_sync_from_qty)
            # with col_a:
            #     st.number_input("Amount ($)",
            #                     min_value=0.0, step=0.01, format="%.2f",
            #                     key=amt_key, on_change=_sync_from_amt)
            #
            # # confirm button
            # if st.button("Confirm purchase", key=f"buy-{r.Ticker}"):
            #     qty = st.session_state[qty_key]
            #     amt = st.session_state[amt_key]
            #     st.success(f"Queued order: {qty:,.4f} × {r.Ticker}  (≈ ${amt:,.2f})")


    # --------- PAGINATION buttons -----------------------------------------
    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.button("◀︎ Prev", disabled=q.page == 1):
            st.session_state.page = q.page - 1
            st.rerun()
    with next_col:
        # disable “Next” when fewer than page_size rows
        if st.button("Next ▶︎", disabled=len(page_df) < 10):
            st.session_state.page = q.page + 1
            st.rerun()
