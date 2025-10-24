import re
import streamlit as st
import markdown

from chat import answer_question
from quiz import generate_quiz, grade_quiz
from process_video import generate_structure

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Video Content Analyzer", layout="wide")

# --------------------------------------------------
# Session defaults
# --------------------------------------------------
def _init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

_init_state("ready", False)
_init_state("structure", {})
_init_state("chunks", [])            # RAG KB
_init_state("chat_history", [])
_init_state("quiz", [])              # list of question dicts
_init_state("quiz_answers", [])      # user answers
_init_state("quiz_feedback", "")     # markdown feedback string

# --------------------------------------------------
# Helpers
# --------------------------------------------------
youtube_pattern1 = r"https?://(www\.)?youtube\.com/watch\?v=[\w-]+"
youtube_pattern2 = r"https?://youtu\.be/[\w-]+"

def build_details_html(node: dict, level: int = 0) -> str:
    """Recursively build nested <details> accordion with visual indent."""
    header = node.get("header", "Untitled")
    raw_md = node.get("content", "").strip()

    content_html = markdown.markdown(
        raw_md, extensions=["fenced_code", "tables", "nl2br", "sane_lists"]
    )

    indent_px = level * 18
    parts = [
        f'<details style="margin-left:{indent_px}px;">',
        f'  <summary><b>{header}</b></summary>',
    ]
    if content_html:
        parts.append(f'  <div>{content_html}</div>')
    for sub in node.get("subsections", []):
        parts.append(build_details_html(sub, level + 1))
    parts.append("</details>")
    return "\n".join(parts)

# -------------------
# Landing page
# -------------------
def landing_page():
    st.title("Video Content Analyzer")
    url = st.text_input("YouTube video URL", key="input_url")

    if url and (re.match(youtube_pattern1, url) or re.match(youtube_pattern2, url)):
        st.video(url)

    if st.button("Process Video"):
        if not (re.match(youtube_pattern1, url) or re.match(youtube_pattern2, url)):
            st.error("Invalid YouTube URL.")
            return
        try:
            with st.spinner("Processing – please wait…"):
                st.session_state.structure, st.session_state.chunks = generate_structure(
                    url
                )
            st.session_state.ready = True
            st.rerun()
        except Exception as e:
            st.error(str(e))

# --------------------------
# Workspace
# ---------------------------------
def workspace():
    if st.button("Back"):
        st.session_state.ready = False
        st.rerun()

    col1, col2 = st.columns([2, 1])

    # ---------- LEFT COLUMN -------
    with col1:
        st.header("Content Structure")
        html = "\n".join(
            build_details_html(sec)
            for sec in st.session_state.structure.get("sections", [])
        )
        st.markdown(html, unsafe_allow_html=True)

        st.divider()

        # -------- Quiz Generator ----
        st.subheader("Quiz")
        q_num = st.number_input("Number of questions", 1, 15, 6)
        if st.button("Generate quiz"):
            try:
                st.session_state.quiz = generate_quiz(
                    st.session_state.structure, q_num
                )
                st.session_state.quiz_answers = ["" for _ in st.session_state.quiz]
                st.session_state.quiz_feedback = ""
                st.rerun()
            except Exception as e:
                st.error(f"Quiz error: {e}")

        # --- Display Quiz ---
        if st.session_state.quiz:
            st.markdown("#### Questions")
            for i, q in enumerate(st.session_state.quiz):
                st.markdown(f"**{i + 1}. {q['question']}**")

                # Single-answer -> radio ; Multiple-answer -> multiselect
                if len(q["answers"]) == 1:
                    st.session_state.quiz_answers[i] = st.radio(
                        "", options=q["choices"], key=f"qa_{i}", index=None
                    )
                else:
                    sel = st.multiselect("", options=q["choices"], key=f"qa_{i}")
                    st.session_state.quiz_answers[i] = ",".join(sel)

            if st.button("Submit answers"):
                _, fb = grade_quiz(
                    st.session_state.quiz,
                    st.session_state.quiz_answers,
                    st.session_state.chunks,
                )
                st.session_state.quiz_feedback = fb
                st.rerun()

            if st.session_state.quiz_feedback:
                st.markdown(st.session_state.quiz_feedback)

    # ---------- RIGHT COLUMN ----------
    with col2:
        st.header("Chat")
        for role, msg in st.session_state.chat_history:
            st.markdown(f"**{role.title()}:** {msg}")

        q = st.text_input("Ask", key="chat_in")
        if st.button("Send", key="send_btn"):
            st.session_state.chat_history.append(("user", q))
            try:
                ans = answer_question(q, st.session_state.chunks)
            except Exception as e:
                ans = f"Error: {e}"
            st.session_state.chat_history.append(("assistant", ans))
            st.rerun()

if st.session_state.ready:
    workspace()
else:
    landing_page()
