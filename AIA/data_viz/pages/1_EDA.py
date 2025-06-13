import numpy as np
import streamlit as st
import plotly.express as px
from utils.load_data import load_dataset

# ensure the dataframe is loaded
if "df" not in st.session_state:
    st.session_state["df"] = load_dataset()
df = st.session_state["df"]

st.header("Exploratory data analysis")

countries = st.multiselect("Country", sorted(df["Country"].unique()))
sources   = st.multiselect("Isolation source", sorted(df["Isolation_Source"].unique()))

mask_country = df["Country"].isin(countries) if countries else np.ones(len(df), dtype=bool)
mask_source  = df["Isolation_Source"].isin(sources) if sources else np.ones(len(df), dtype=bool)
sub = df[mask_country & mask_source]

def bar(data, x, title):
    fig = px.bar(data, x=x, y="n", title=title)
    fig.update_layout(yaxis_title=None)
    return fig

st.plotly_chart(
    bar(sub.groupby("Country").size().reset_index(name="n"), "Country", "Samples per country"),
    use_container_width=True
)
st.plotly_chart(
    bar(sub.groupby("Isolation_Source").size().reset_index(name="n"), "Isolation_Source", "Samples per isolation source"),
    use_container_width=True
)
st.plotly_chart(
    bar(sub.groupby("Fungi_species").size().reset_index(name="n"), "Fungi_species", "Samples per species")
        .update_layout(xaxis_tickangle=45),
    use_container_width=True
)

st.subheader("Filtered table")
st.dataframe(sub)
