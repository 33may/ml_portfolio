import streamlit as st

from ui_portfolio import portfolio_page
from ui_stock_list import stock_list

st.set_page_config(
    page_title="Portfolio Data Visualization",
    layout="wide",
    initial_sidebar_state="expanded"
)

tab1, tab2 = st.tabs(["Stocks List", "Portfolio"])

with tab1:
    st.header("Stocks List")
    stock_list()

with tab2:
    st.header("Portfolio Data")
    portfolio_page()
