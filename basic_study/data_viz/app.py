import streamlit as st
from utils.load_data import load_dataset

st.set_page_config(page_title="Fungi DNA Explorer", layout="wide")
st.title("Fungi DNA Explorer")

if "df" not in st.session_state:
    st.session_state["df"] = load_dataset()

st.write("This explorer takes the raw table of 900 fungal DNA records, cleans and inspects it, and then walks the data all the way to interactive visuals and gentle sonification.")
st.write("First the CSV is ingested and a quick descriptive pass counts samples by country, isolation source and species; that early scan confirms sequence lengths and reveals basic distribution quirks. Next every sequence is written to FASTA, aligned with MAFFT and passed to FastTree to obtain a rough maximum-likelihood phylogeny. The aligned matrix is turned into numbers—each column one-hot-encoded for A, C, G, T or gap—then UMAP compresses columns into three smooth coordinates, giving each sample an (x, y, z) point that reflects overall genetic similarity. Those coordinates are appended to the master table and saved as *fungi\_with\_coords.csv*, which all Streamlit pages load from cache.")

st.write("Inside the app the **EDA** view lets you filter and chart the metadata on the fly; the **Sequence map** page shows any individual genome as a pastel colour bar and renders a calm sine-wave interpretation of its bases; the **3-D clusters** view plots the UMAP embedding so you can spin, zoom and recolour the genetic landscape by genus, country or source. Everything you see—counts, stripes, audio and 3-d points—derives from the single, reproducible pipeline that starts with the original semicolon-separated CSV and ends with a responsive dashboard.")
