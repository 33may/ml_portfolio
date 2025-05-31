import streamlit as st

from ui_stock_list import stock_list

st.set_page_config(
    page_title="Финансовое приложение",
    layout="wide",
    initial_sidebar_state="expanded"
)

tab1, tab2, tab3 = st.tabs(["Список акций", "График портфеля", "О приложении"])

with tab1:
    st.header("Первая вкладка: список акций")
    stock_list()

with tab2:
    st.header("Вторая вкладка: график портфеля")
    # Здесь будет код графика портфеля...

with tab3:
    st.header("Третья вкладка: информация об приложении")
    st.write("Здесь можно разместить описание и документацию.")
