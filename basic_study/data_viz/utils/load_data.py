import pandas as pd
import streamlit as st

@st.cache_data
def load_dataset(path: str = "fungi_with_coords.csv") -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    df["genus"] = df["Fungi_species"].str.split().str[0]
    return df
