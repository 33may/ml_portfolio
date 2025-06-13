import math
import wave
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from io import BytesIO

if "df" not in st.session_state:
    st.session_state["df"] = pd.read_csv("fungi_with_coords.csv", sep=";")
df = st.session_state["df"]

cmap = colormaps.get_cmap("Pastel1")
NUC_COL = {
    "A": cmap(0)[:3],
    "C": cmap(1)[:3],
    "G": cmap(2)[:3],
    "T": cmap(3)[:3],
    "-": (1, 1, 1)
}

def seq_to_rgb(seq):
    arr = np.array([NUC_COL.get(b.upper(), (0, 0, 0)) for b in seq])
    return arr.reshape(1, -1, 3)

NOTE_MAP = {"A": 48, "C": 50, "G": 55, "T": 57}

def pitch_to_freq(pitch):
    return 440.0 * (2 ** ((pitch - 69) / 12))

def dna_to_wav(seq, bpm=60, sr=44100):
    note_sec = 0.5 * 60 / bpm
    n_note = int(sr * note_sec)
    seq = [b for b in seq.upper() if b in NOTE_MAP]
    if not seq:
        return None
    audio = np.zeros(len(seq) * n_note)
    for i, base in enumerate(seq):
        f0 = pitch_to_freq(NOTE_MAP[base])
        f1 = f0 if i == len(seq) - 1 else pitch_to_freq(NOTE_MAP[seq[i + 1]])
        glide = np.linspace(f0, f1, n_note)
        t = np.arange(n_note) / sr
        chunk = 0.3 * np.sin(2 * math.pi * glide * t)
        env = np.linspace(0, 1, int(0.05 * n_note))
        chunk[:len(env)] *= env
        chunk[-len(env):] *= env[::-1]
        audio[i * n_note : (i + 1) * n_note] += chunk
    audio /= np.max(np.abs(audio) + 1e-9)
    buf = BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    buf.seek(0)
    return buf

def rgb_to_hex(rgb):
    r, g, b = (int(255 * x) for x in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"

st.title("Sequence colour map and calm sonification")

idx = st.selectbox("Sample index", df.index)
row = df.loc[idx]

st.write(
    f"Species: {row['Fungi_species']} | "
    f"Country: {row['Country']} | "
    f"Isolation source: {row['Isolation_Source']}"
)

height_px = 240
img = seq_to_rgb(row["Sequence"])
fig_w = len(row["Sequence"]) / 40
fig_h = height_px / 100
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.imshow(img, aspect="auto")
ax.axis("off")
st.pyplot(fig)

cols = st.columns(4)
for col_idx, nuc in enumerate(["A", "C", "G", "T"]):
    hex_color = rgb_to_hex(NUC_COL[nuc])
    with cols[col_idx]:
        st.markdown(
            f"<div style='display:flex; align-items:center;'>"
            f"<div style='width:20px;height:20px;background:{hex_color};border:1px solid #000;'></div>"
            f"<span style='margin-left:6px;'>{nuc}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

wav_buf = dna_to_wav(row["Sequence"][:400])
if wav_buf:
    st.audio(wav_buf.getvalue(), format="audio/wav")
else:
    st.error("No audible bases in this sequence.")
