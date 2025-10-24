import streamlit as st
import plotly.express as px
from utils.load_data import load_dataset

from utils.pallete import GENUS_COLORS, SOURCE_COLORS

if "df" not in st.session_state:
    st.session_state["df"] = load_dataset()
df = st.session_state["df"]

st.header("3-D embedding")

colour_by = st.selectbox("Colour points by", ["genus", "Country", "Isolation_Source"])
colour_map = {
    "genus": GENUS_COLORS,
    "Isolation_Source": SOURCE_COLORS
}.get(colour_by, None)

fig = px.scatter_3d(
    df,
    x="x3d", y="y3d", z="z3d",
    color=df[colour_by],
    hover_name="Fungi_species",
    hover_data={"Country": True, "Isolation_Source": True},
    color_discrete_map=colour_map
)

fig.update_traces(marker=dict(size=4))
fig.update_layout(
    width=1000,      # desired pixel width
    height=800,      # desired pixel height
    margin=dict(l=0, r=0, b=0, t=40)
)

st.plotly_chart(fig, use_container_width=False)
