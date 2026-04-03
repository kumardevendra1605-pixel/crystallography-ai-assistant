import time
import base64
import os
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(page_title="xTal.ai", page_icon="🔷", layout="wide", initial_sidebar_state="collapsed")

from src.qa_parser import load_all_qa
from src.embeddings import build_and_cache_embeddings
from src.vector_store import create_vector_store
from src.rag_pipeline import retrieve_and_respond

# ── Viewport meta + mobile viewport fix ──────────────────────────────────────
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
""", unsafe_allow_html=True)

# ── Encode background image ───────────────────────────────────────────────────
with open("assets/1212.png", "rb") as _f:
    _BG = base64.b64encode(_f.read()).decode()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}

/* ── Full-app background ── */
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{_BG}");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: #ececec;
}}

/* ── Transparent layers so bg shows through ── */
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stMain"],
[data-testid="stAppViewBlockContainer"] {{
    background: transparent !important;
}}

/* ── Main content: full width on mobile, capped on desktop ── */
[data-testid="stMainBlockContainer"] {{
    max-width: 100% !important;
    padding: 0 12px 80px 12px !important;
}}
@media (min-width: 768px) {{
    [data-testid="stMainBlockContainer"] {{
        max-width: 100% !important;
        padding: 0 24px 80px 24px !important;
    }}
}}

/* ── Hide Streamlit chrome we don't need ── */
#MainMenu, footer {{ visibility: hidden; }}
footer {{ display: none !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}
[data-testid="stFooter"] {{ display: none !important; }}
.viewerBadge_container__r5tak,
.viewerBadge_link__qRIco,
#stDecoration {{ display: none !important; }}

/* ── Keep header/toolbar visible and on top ── */
[data-testid="stHeader"] {{
    background: transparent !important;
    z-index: 100 !important;
}}
[data-testid="stToolbar"] {{
    z-index: 101 !important;
}}

/* ── Sidebar — full-width overlay on mobile ── */
[data-testid="stSidebar"] {{
    background: rgba(10, 10, 10, 0.97) !important;
    border-right: 1px solid #1e1e1e !important;
    min-width: 260px !important;
    max-width: 80vw !important;
}}
@media (max-width: 767px) {{
    [data-testid="stSidebar"] {{
        min-width: 100vw !important;
        max-width: 100vw !important;
    }}
}}
[data-testid="stSidebar"] > div:first-child {{
    padding-top: 12px !important;
}}

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] .stButton > button {{
    background: transparent !important;
    border: none !important;
    color: #c8c8c8 !important;
    text-align: left !important;
    padding: 10px 10px !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    width: 100% !important;
    transition: background 0.18s ease, color 0.18s ease !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    font-family: 'Inter', sans-serif !important;
    min-height: 44px !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(255,255,255,0.06) !important;
    color: #ffffff !important;
}}

/* New chat button */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #ececec !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    margin-bottom: 4px !important;
    padding: 12px 14px !important;
    min-height: 48px !important;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
    background: rgba(255,255,255,0.09) !important;
    border-color: rgba(255,255,255,0.18) !important;
}}

/* Active conversation */
.active-conv .stButton > button,
.conv-row.active-conv .stButton > button {{
    background: rgba(74,158,255,0.12) !important;
    border-left: 2px solid #4a9eff !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    padding-left: 8px !important;
}}

/* Delete button — touch-friendly */
.del-btn .stButton > button,
[data-testid="stHorizontalBlock"] > div:last-child .stButton > button {{
    color: #555 !important;
    font-size: 14px !important;
    padding: 0 !important;
    margin: 0 auto !important;
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    min-height: 36px !important;
    max-width: 36px !important;
    max-height: 36px !important;
    line-height: 1 !important;
    border-radius: 6px !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}}
[data-testid="stHorizontalBlock"] > div:last-child .stButton > button:hover,
[data-testid="stHorizontalBlock"] > div:last-child .stButton > button:active {{
    color: #ff6b6b !important;
    background: rgba(255,80,80,0.1) !important;
}}

/* Ensure horizontal block aligns items to center vertically */
[data-testid="stHorizontalBlock"] {{
    align-items: center !important;
}}
[data-testid="stHorizontalBlock"] > div {{
    display: flex !important;
    align-items: center !important;
}}

/* Section labels */
.section-label {{
    font-size: 11px;
    font-weight: 600;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    padding: 10px 12px 3px 12px;
}}

/* Sidebar divider */
.sidebar-divider {{
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 8px 0;
}}

/* Empty sidebar state */
.empty-state {{
    text-align: center;
    padding: 32px 16px;
    color: #3a3a3a;
    font-size: 14px;
}}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {{
    background: rgba(0,0,0,0.45) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
    margin-bottom: 8px !important;
    backdrop-filter: blur(8px);
    word-break: break-word !important;
    overflow-wrap: break-word !important;
}}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {{
    font-size: 15px !important;
    line-height: 1.6 !important;
}}

/* ── Chat input — sticky at bottom on mobile ── */
[data-testid="stBottom"] {{
    position: sticky !important;
    bottom: 0 !important;
    background: transparent !important;
    padding: 8px 0 env(safe-area-inset-bottom, 8px) !important;
    z-index: 50 !important;
}}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:active {{
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    color: #ececec !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 16px !important; /* prevents iOS zoom on focus */
}}
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div:focus-within {{
    background: rgba(0,0,0,0.55) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: none !important;
}}

/* ── Expanders ── */
[data-testid="stExpander"] {{
    background: rgba(10,10,10,0.7) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(8px);
}}
[data-testid="stExpander"] summary {{
    font-size: 14px !important;
    padding: 10px 14px !important;
    min-height: 44px !important;
    display: flex !important;
    align-items: center !important;
}}

/* ── Divider ── */
hr {{
    border-color: rgba(255,255,255,0.06) !important;
}}

/* ── Spinner ── */
[data-testid="stSpinner"] {{
    color: #4a9eff !important;
}}

/* ── Info box ── */
[data-testid="stInfo"] {{
    background: rgba(74,158,255,0.1) !important;
    border: 1px solid rgba(74,158,255,0.2) !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}}

/* ── Prevent horizontal scroll on mobile ── */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {{
    overflow-x: hidden !important;
}}

/* ── Responsive text ── */
@media (max-width: 480px) {{
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {{
        font-size: 14px !important;
    }}
    .section-label {{
        font-size: 10px;
    }}
}}
</style>
""", unsafe_allow_html=True)

# ── Floating sidebar toggle ───────────────────────────────────────────────────
import streamlit.components.v1 as _cv1
_cv1.html("""
<style>
#st-toggle {
    position: fixed;
    top: 10px; left: 10px;
    z-index: 999999;
    width: 40px; height: 40px;
    background: rgba(20,20,20,0.88);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    color: #ccc;
    font-size: 18px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.2s, border-color 0.2s;
    backdrop-filter: blur(10px);
    -webkit-tap-highlight-color: transparent;
    touch-action: manipulation;
}
#st-toggle:hover,
#st-toggle:active { background: rgba(40,40,40,0.95); border-color: rgba(255,255,255,0.2); color: #fff; }
</style>
<button id="st-toggle" title="Toggle sidebar" onclick="toggle()">&#9776;</button>
<script>
function toggle() {
    var d = window.parent.document;
    var btn = d.querySelector('[data-testid="collapsedControl"] button')
           || d.querySelector('button[aria-label="Close sidebar"]')
           || d.querySelector('button[aria-label="Open sidebar"]');
    if (btn) { btn.click(); return; }
    var all = d.querySelectorAll('button');
    for (var b of all) {
        var r = b.getBoundingClientRect();
        if (r.left < 80 && r.top < 100 && r.width < 60) { b.click(); return; }
    }
}
</script>
""", height=0)

# ── Pipeline ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def setup_pipeline():
    from src.query_processor import build_word_vocab
    qa_pairs = load_all_qa(os.getenv("DATA_DIR", "data/docx"))
    build_word_vocab(qa_pairs)
    embeddings = build_and_cache_embeddings(qa_pairs)
    index = create_vector_store(embeddings)
    return index, qa_pairs

index, qa_pairs = setup_pipeline()

# ── Session state ─────────────────────────────────────────────────────────────
if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "active_id" not in st.session_state:
    st.session_state.active_id = None
if "rename_id" not in st.session_state:
    st.session_state.rename_id = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def _new_conversation() -> str:
    cid = str(int(time.time() * 1000))
    st.session_state.conversations.insert(0, {
        "id": cid, "title": "New conversation",
        "messages": [], "created_at": time.time(), "titled": False,
    })
    st.session_state.active_id = cid
    return cid

def _get_active() -> dict | None:
    for c in st.session_state.conversations:
        if c["id"] == st.session_state.active_id:
            return c
    return None

def _delete_conversation(cid: str):
    st.session_state.conversations = [c for c in st.session_state.conversations if c["id"] != cid]
    if st.session_state.active_id == cid:
        st.session_state.active_id = (
            st.session_state.conversations[0]["id"] if st.session_state.conversations else None
        )

def _generate_title(query: str) -> str:
    import re
    stop = {
        "what","how","why","when","where","which","who","is","are","was","were",
        "the","a","an","in","on","at","to","for","of","and","or","but","not",
        "with","this","that","can","does","do","will","would","could","should",
        "have","has","had","be","been","being","about","from","into","than",
        "then","just","also","some","more","very","any","all","its","it","my",
        "your","our","their","we","you","i","me","us","them","he","she","they",
        "please","tell","explain","give",
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    kws = [w for w in words if w not in stop]
    if not kws:
        return query[:45] + ("…" if len(query) > 45 else "")
    return " ".join(w.title() for w in kws[:5])[:50]

def _group_conversations(convs):
    now = datetime.now()
    today, yesterday = now.date(), now.date() - timedelta(days=1)
    week_ago = now.date() - timedelta(days=7)
    groups = {"Today": [], "Yesterday": [], "Previous 7 Days": [], "Older": []}
    for c in convs:
        d = datetime.fromtimestamp(c.get("created_at", 0)).date()
        if d == today: groups["Today"].append(c)
        elif d == yesterday: groups["Yesterday"].append(c)
        elif d >= week_ago: groups["Previous 7 Days"].append(c)
        else: groups["Older"].append(c)
    return {k: v for k, v in groups.items() if v}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("＋  New conversation", use_container_width=True, type="primary"):
        _new_conversation()
        st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    if not st.session_state.conversations:
        st.markdown('<div class="empty-state">No conversations yet</div>', unsafe_allow_html=True)
    else:
        for group_label, convs in _group_conversations(st.session_state.conversations).items():
            st.markdown(f'<div class="section-label">{group_label}</div>', unsafe_allow_html=True)
            for conv in convs:
                is_active = conv["id"] == st.session_state.active_id

                if st.session_state.rename_id == conv["id"]:
                    new_title = st.text_input("", value=conv["title"],
                        key=f"ri_{conv['id']}", label_visibility="collapsed")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✓ Save", key=f"rok_{conv['id']}"):
                            conv["title"] = new_title.strip() or conv["title"]
                            st.session_state.rename_id = None
                            st.rerun()
                    with c2:
                        if st.button("✕ Cancel", key=f"rcancel_{conv['id']}"):
                            st.session_state.rename_id = None
                            st.rerun()
                    continue

                is_active = conv["id"] == st.session_state.active_id

                col_t, col_d = st.columns([10, 1], gap="small", vertical_alignment="center")
                with col_t:
                    st.markdown(f'<div class="{"active-conv" if is_active else ""}">', unsafe_allow_html=True)
                    if st.button(conv["title"], key=f"c_{conv['id']}", use_container_width=True):
                        st.session_state.active_id = conv["id"]
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                with col_d:
                    if st.button("✕", key=f"d_{conv['id']}"):
                        _delete_conversation(conv["id"])
                        st.rerun()

# ── Ensure active conversation ────────────────────────────────────────────────
if st.session_state.active_id is None or _get_active() is None:
    _new_conversation()
active = _get_active()

# ── Result rendering ──────────────────────────────────────────────────────────
def _conf_color(s):
    return "#2ecc71" if s >= 0.75 else "#f39c12" if s >= 0.50 else "#e74c3c"

def render_result(result: dict):
    conf = result["confidence"]
    sources = result.get("sources", [])
    alternatives = result.get("alternatives", [])

    if result.get("matched_question"):
        st.markdown(
            f'<small style="color:#888">Confidence: '
            f'<span style="color:{_conf_color(conf)};font-weight:600">{conf:.0%}</span>'
            f' &nbsp;·&nbsp; <i>{result["matched_question"][:80]}</i></small>',
            unsafe_allow_html=True,
        )
    if result["query_info"].get("corrected"):
        st.caption(f'🔤 Interpreted as: *"{result["query_info"]["corrected"]}"*')
    if result.get("did_you_mean") and result.get("clarification_needed"):
        st.info(f'💡 Did you mean: *"{result["did_you_mean"]}"*?')
    if sources:
        with st.expander(f"📚 {len(sources)} source{'s' if len(sources)!=1 else ''} used"):
            for src in sources:
                src_conf = src['confidence']
                st.markdown(
                    f"**{src['citation']}** "
                    f"<span style='color:{_conf_color(src_conf)}'>"
                    f"({src_conf:.0%})</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Q:** {src['question']}")
                st.markdown(f"**A:** {src['answer']}")
                st.divider()
    if alternatives:
        with st.expander(f"🔗 {len(alternatives)} related question{'s' if len(alternatives)!=1 else ''}"):
            for alt in alternatives:
                st.markdown(f"- {alt['question']}  \n  *{alt['citation']} · {alt['confidence']:.0%}*")

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in active["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("result"):
            render_result(msg["result"])

# ── Empty state ───────────────────────────────────────────────────────────────
if not active["messages"]:
    st.markdown("""
    <div style="
        text-align:center;
        padding: 32px 16px 16px;
        color: rgba(255,255,255,0.35);
        font-size: 15px;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.01em;
    ">
        Ask anything about crystallography to get started.
    </div>
    """, unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask anything about crystallography...")

if prompt:
    is_first = not active.get("titled")
    active["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner(f"Searching {len(qa_pairs)} Q&A pairs..."):
            result = retrieve_and_respond(prompt, index, qa_pairs)
        st.markdown(result["answer"])
        render_result(result)
    active["messages"].append({"role": "assistant", "content": result["answer"], "result": result})
    if is_first:
        active["title"] = _generate_title(prompt)
        active["titled"] = True
        st.rerun()
