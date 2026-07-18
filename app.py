# app.py
# KBSeek AI — Streamlit UI for IT Service Desk agents
# Run with: streamlit run app.py

import html
import streamlit as st
from search import ask_kbseek

# ── PAGE CONFIG ──────────────────────────────────────
# This MUST be the very first Streamlit call in the file.
# layout="centered" keeps the text readable on wide monitors.
st.set_page_config(
    page_title="KBSeek AI",
    page_icon="🔍",
    layout="wide"
)

# ── CUSTOM CSS ───────────────────────────────────────
# Streamlit lets us inject raw CSS for styling elements
# that its built-in components don't cover.
# unsafe_allow_html=True is required whenever we use raw HTML/CSS.
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }

    /* Left column sticks in place while page scrolls */
    [data-testid="column"]:first-child {
        position: sticky;
        top: 60px;
        align-self: flex-start;
        max-height: calc(100vh - 80px);
        overflow-y: auto;
    }

    /* Right column is its own scrollable panel */
    .right-panel {
        height: calc(100vh - 160px);
        overflow-y: auto;
        padding-right: 6px;
    }

    .source-card {
        background: #ffffff;
        border: 1px solid #d0ddef;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }

    .rrf-pill {
        background: #e4edf8;
        color: #1a4777;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        padding: 2px 9px;
        border-radius: 10px;
    }

    .source-card a {
        color: #1a6bb5;
        font-size: 0.85rem;
        text-decoration: none;
    }
    .source-card a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────
st.markdown("## 🔍 KBSeek AI")
st.caption("IT Service Desk · Knowledge Base Assistant — type the caller's problem below")
st.divider()

# ── SEARCH FORM ──────────────────────────────────────
# st.form groups the text box and button together.
# Without a form, Streamlit would re-run the whole script
# on EVERY keystroke — which would trigger a new search
# each time the agent types a letter. The form prevents that:
# it only re-runs when the agent presses Enter or clicks the button.
with st.form("search_form", clear_on_submit=False):
    query = st.text_input(
        label="Caller's problem:",
        placeholder="e.g. caller forgot their password and can't log in",
    )
    submitted = st.form_submit_button(
        "Search Knowledge Base",
        type="primary",
        use_container_width=True
    )

# ── SEARCH & DISPLAY RESULTS ─────────────────────────
# This block only runs when the agent submits the form.
if submitted:

    if not query.strip():
        st.warning("Please describe the caller's problem before searching.")

    else:
        with st.spinner("Searching knowledge base..."):
            try:
                from search import hybrid_search, generate_answer
                all_results = hybrid_search(query, top_k=20)
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.info(
                    "Make sure Ollama is running (`ollama serve` in a terminal) "
                    "and that you have indexed your KB (`python ingest.py`)."
                )
                st.stop()

        left_col, right_col = st.columns([3, 2])

        # ── RIGHT: render 20 articles immediately after search ──
        with right_col:
            st.markdown("### Related Articles")
            cards = ""
            for i, result in enumerate(all_results):
                title = result.get("metadata", {}).get("title", "Untitled")
                url   = result.get("metadata", {}).get("url", "#")
                score = result.get("rrf_score", "—")
                cards += f"""
                <div class="source-card">
                    <strong>#{i + 1}</strong>&nbsp;<span class="rrf-pill">RRF {score}</span><br>
                    <a href="{html.escape(url)}" target="_blank">{html.escape(title)}</a>
                </div>"""
            st.markdown(f'<div class="right-panel">{cards}</div>', unsafe_allow_html=True)

        # ── LEFT: call LLM now (right column already visible) ───
        with left_col:
            st.markdown("### Answer")
            answer_box  = st.empty()
            try:
                answer_stream, _ = generate_answer(query, all_results[:3])
            except Exception as e:
                st.error(f"Answer generation failed: {e}")
                st.stop()
            full_answer = ""
            for token in answer_stream:
                full_answer += token
                answer_box.info(full_answer + "▌")
            answer_box.info(full_answer)

        st.divider()
        st.caption(f'Query: "{query}"')
