from __future__ import annotations
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

# ── Encode background images ──────────────────────────────────────────────────
with open("assets/1212.png", "rb") as _f:
    _BG_DESKTOP = base64.b64encode(_f.read()).decode()
with open("assets/1111.png", "rb") as _f:
    _BG_MOBILE = base64.b64encode(_f.read()).decode()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body {{
    margin: 0;
    padding: 0;
    overflow-x: hidden !important;
    width: 100% !important;
}}

/* ── Mobile background (default) ── */
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{_BG_MOBILE}");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: scroll;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: #ececec;
    overflow-x: hidden !important;
    width: 100% !important;
    max-width: 100vw !important;
}}

/* ── Desktop background ── */
@media (min-width: 768px) {{
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{_BG_DESKTOP}");
        background-attachment: fixed;
    }}
}}

/* ── Transparent layers so bg shows through ── */
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stMain"],
[data-testid="stAppViewBlockContainer"] {{
    background: transparent !important;
}}

/* ── Main content area: always fills available width ── */
[data-testid="stMain"] {{
    flex: 1 1 0% !important;
    overflow-x: hidden !important;
}}
[data-testid="stMainBlockContainer"] {{
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 12px 80px 12px !important;
    overflow-x: hidden !important;
}}
@media (min-width: 768px) {{
    [data-testid="stMainBlockContainer"] {{
        padding: 0 24px 80px 24px !important;
    }}
}}

/* ────────────────────────────────────────────────────────────
   MOBILE LAYOUT — everything below 768 px
   The critical goal: sidebar is completely off-canvas (hidden
   to the left) by default, main content = 100 vw, no margin.
   ──────────────────────────────────────────────────────────── */
@media (max-width: 767px) {{

    /* 1. Prevent the app-level flex row from overflowing */
    [data-testid="stAppViewContainer"] {{
        display: flex !important;
        flex-direction: row !important;
        overflow: hidden !important;
        width: 100vw !important;
        max-width: 100vw !important;
        position: relative !important;
    }}

    /* 2. Sidebar: fixed overlay, completely hidden off-canvas by default */
    [data-testid="stSidebar"] {{
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        min-width: 100vw !important;
        max-width: 100vw !important;
        height: 100vh !important;
        z-index: 999998 !important;
        transform: translateX(-100%) !important;
        transition: transform 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;
        will-change: transform !important;
        overflow-y: auto !important;
        /* Flatten the Streamlit sidebar width allocation to 0 in flow */
        flex-shrink: 0 !important;
        flex-basis: 0px !important;
    }}

    /* 3. Main section: take up the full viewport, zero left margin */
    [data-testid="stMain"],
    section.main,
    [data-testid="stAppViewContainer"] > section:not([data-testid="stSidebar"]) {{
        margin-left: 0 !important;
        padding-left: 0 !important;
        width: 100vw !important;
        min-width: 100vw !important;
        max-width: 100vw !important;
        flex: 1 1 100% !important;
        position: relative !important;
        left: 0 !important;
        transform: none !important;
        overflow-x: hidden !important;
    }}

    /* 4. Kill any residual horizontal overflow everywhere */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stVerticalBlock"] {{
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }}

    /* 5. Hide Streamlit-native sidebar toggle controls on mobile */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarNavItems"],
    [data-testid="stSidebarUserContent"] > div:first-child {{
        /* don't hide user content, only native controls */
    }}
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarNavItems"] {{
        display: none !important;
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

/* Hide Streamlit's default sidebar collapse arrow — we use our own toggle */
[data-testid="collapsedControl"] {{ display: none !important; }}

/* Hide the sidebar toggle button that appears in the header */
[data-testid="stSidebarNavItems"] {{ display: none !important; }}
button[kind="header"] {{ display: none !important; }}
[data-testid="stHeader"] button:first-child {{ display: none !important; }}

/* ── Keep header/toolbar visible and on top ── */
[data-testid="stHeader"] {{
    background: transparent !important;
    z-index: 100 !important;
}}
[data-testid="stToolbar"] {{
    z-index: 101 !important;
}}

/* ── Sidebar — desktop appearance ── */
[data-testid="stSidebar"] {{
    background: rgba(10, 10, 10, 0.97) !important;
    border-right: 1px solid #1e1e1e !important;
    min-width: 260px !important;
    max-width: 80vw !important;
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
    width: 100% !important;
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

/* ── "Created by" label — fixed below chat input ── */
.created-by {{
    position: fixed;
    bottom: 10px;
    left: 0; right: 0;
    text-align: center;
    color: rgba(255,255,255,0.45);
    font-size: 12px;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.04em;
    pointer-events: none;
    z-index: 9999;
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
st.markdown("""
<style>
/* ───────────────────────────────────────────────────────────────
   HAMBURGER TOGGLE BUTTON
   Uses .st-key-sidebar_toggle — the class Streamlit auto-generates
   on the wrapper of every element whose key="sidebar_toggle".
   This is the only reliable way to target a specific st.button().
─────────────────────────────────────────────────────────────── */

/* Desktop: thin vertical tab on the left edge */
.st-key-sidebar_toggle {
    position: fixed !important;
    top: 50% !important;
    left: 0 !important;
    transform: translateY(-50%) !important;
    z-index: 1000000 !important;
    width: 22px !important;
    height: 64px !important;
    margin: 0 !important;
    padding: 0 !important;
}
.st-key-sidebar_toggle button {
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    padding: 0 !important;
    background: rgba(18,18,18,0.85) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-left: none !important;
    border-radius: 0 10px 10px 0 !important;
    color: #aaa !important;
    font-size: 14px !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    -webkit-tap-highlight-color: transparent !important;
    touch-action: manipulation !important;
    transition: width 0.2s ease, background 0.2s ease, color 0.2s ease !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.st-key-sidebar_toggle button:hover,
.st-key-sidebar_toggle button:active {
    background: rgba(40,40,40,0.97) !important;
    color: #fff !important;
}

/* Mobile: square button, top-left corner with 14px inset */
@media (max-width: 767px) {
    .st-key-sidebar_toggle {
        top: 14px !important;
        left: 14px !important;
        transform: none !important;
        width: 44px !important;
        height: 44px !important;
    }
    .st-key-sidebar_toggle button {
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        font-size: 20px !important;
        background: rgba(15,15,15,0.88) !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.45) !important;
    }
    .st-key-sidebar_toggle button:hover,
    .st-key-sidebar_toggle button:active {
        background: rgba(35,35,35,0.97) !important;
        color: #fff !important;
    }
}

/* ── Drawer backdrop overlay ── */
#xtal-drawer-backdrop {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.55);
    z-index: 999990;
    opacity: 0;
    pointer-events: none;      /* never intercept clicks when hidden */
    transition: opacity 0.28s ease;
    backdrop-filter: blur(2px);
    -webkit-backdrop-filter: blur(2px);
}

/* ── Mobile: top padding so content clears the hamburger ── */
@media (max-width: 767px) {
    [data-testid="stMainBlockContainer"] {
        padding-top: 70px !important;
    }
}
</style>
<div id="xtal-drawer-backdrop"></div>
""", unsafe_allow_html=True)

if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = False

# Render the hamburger button (no wrapper div needed — we use .st-key-sidebar_toggle)
if st.button("☰", key="sidebar_toggle"):
    st.session_state.sidebar_open = not st.session_state.sidebar_open
    st.rerun()

# Apply sidebar open/close + hamburger positioning via JS
_sidebar_state = "open" if st.session_state.sidebar_open else "closed"
import streamlit.components.v1 as _cv1
_cv1.html(f"""
<script>
(function() {{
    var d = window.parent.document;
    var state = "{_sidebar_state}";

    /* ── Find a button by its visible text content ── */
    function findBtn(text) {{
        var all = d.querySelectorAll('button');
        for (var i = 0; i < all.length; i++) {{
            if (all[i].textContent.trim() === text) return all[i];
        }}
        return null;
    }}

    /* ── Position the hamburger button with position:fixed ──────────────
       CSS selectors like st-key-* are absent in this Streamlit build.
       style.setProperty with 'important' priority always overrides
       emotion CSS class styles, making this 100% reliable.           */
    function positionHamburger() {{
        var btn = findBtn('\u2630');
        if (!btn) {{ setTimeout(positionHamburger, 80); return; }}

        var mobile = window.parent.innerWidth <= 767;

        /* Common properties */
        btn.style.setProperty('position', 'fixed', 'important');
        btn.style.setProperty('z-index', '1000000', 'important');
        btn.style.setProperty('padding', '0', 'important');
        btn.style.setProperty('cursor', 'pointer', 'important');
        btn.style.setProperty('display', 'flex', 'important');
        btn.style.setProperty('align-items', 'center', 'important');
        btn.style.setProperty('justify-content', 'center', 'important');
        btn.style.setProperty('-webkit-backdrop-filter', 'blur(12px)', 'important');
        btn.style.setProperty('backdrop-filter', 'blur(12px)', 'important');
        btn.style.setProperty('-webkit-tap-highlight-color', 'transparent', 'important');
        btn.style.setProperty('touch-action', 'manipulation', 'important');
        btn.style.setProperty('transition', 'background 0.2s ease, color 0.2s ease', 'important');

        if (mobile) {{
            btn.style.setProperty('top',    '14px', 'important');
            btn.style.setProperty('left',   '14px', 'important');
            btn.style.setProperty('right',  'auto', 'important');
            btn.style.setProperty('transform', 'none', 'important');
            btn.style.setProperty('width',  '44px', 'important');
            btn.style.setProperty('height', '44px', 'important');
            btn.style.setProperty('min-width',  '44px', 'important');
            btn.style.setProperty('min-height', '44px', 'important');
            btn.style.setProperty('border-radius', '10px', 'important');
            btn.style.setProperty('border', '1px solid rgba(255,255,255,0.15)', 'important');
            btn.style.setProperty('background', 'rgba(15,15,15,0.92)', 'important');
            btn.style.setProperty('color', '#ffffff', 'important');
            btn.style.setProperty('font-size', '20px', 'important');
            btn.style.setProperty('box-shadow', '0 2px 16px rgba(0,0,0,0.55)', 'important');
        }} else {{
            /* Desktop: thin tab on left edge */
            btn.style.setProperty('top',    '50%',  'important');
            btn.style.setProperty('left',   '0',    'important');
            btn.style.setProperty('right',  'auto', 'important');
            btn.style.setProperty('transform', 'translateY(-50%)', 'important');
            btn.style.setProperty('width',  '22px', 'important');
            btn.style.setProperty('height', '64px', 'important');
            btn.style.setProperty('min-width',  '22px', 'important');
            btn.style.setProperty('min-height', '0', 'important');
            btn.style.setProperty('border-radius', '0 10px 10px 0', 'important');
            btn.style.setProperty('border', '1px solid rgba(255,255,255,0.10)', 'important');
            btn.style.setProperty('border-left', 'none', 'important');
            btn.style.setProperty('background', 'rgba(18,18,18,0.85)', 'important');
            btn.style.setProperty('color', '#aaa', 'important');
            btn.style.setProperty('font-size', '14px', 'important');
            btn.style.setProperty('box-shadow', 'none', 'important');
        }}
    }}

    /* ── Apply sidebar open/close state + reset main content ── */
    function applyMobileSidebar() {{
        if (window.parent.innerWidth > 767) return;

        var sidebar  = d.querySelector('[data-testid="stSidebar"]');
        var main     = d.querySelector('[data-testid="stMain"]');
        var backdrop = d.getElementById('xtal-drawer-backdrop');
        if (!sidebar) {{ setTimeout(applyMobileSidebar, 50); return; }}

        var baseStyle = [
            'position:fixed', 'top:0', 'left:0',
            'width:85vw', 'min-width:280px', 'max-width:360px',
            'height:100vh', 'z-index:999995',
            'transition:transform 0.3s cubic-bezier(0.4,0,0.2,1)',
            'will-change:transform', 'overflow-y:auto', 'overflow-x:hidden',
            'background:rgba(10,10,10,0.98)', 'flex-basis:0px',
            'box-shadow:4px 0 24px rgba(0,0,0,0.6)',
        ].join('!important;') + '!important;';

        if (state === 'open') {{
            sidebar.style.cssText = baseStyle + 'transform:translateX(0)!important;';
            if (backdrop) {{
                backdrop.style.pointerEvents = 'auto';
                backdrop.style.display = 'block';
                void backdrop.offsetWidth;
                backdrop.style.opacity = '1';
            }}
            d.body.style.overflow = 'hidden';
        }} else {{
            sidebar.style.cssText = baseStyle + 'transform:translateX(-100%)!important;';
            if (backdrop) {{
                backdrop.style.opacity = '0';
                backdrop.style.pointerEvents = 'none';
                setTimeout(function() {{ backdrop.style.display = 'none'; }}, 300);
            }}
            d.body.style.overflow = '';
        }}

        if (main) {{
            main.style.cssText = [
                'margin-left:0', 'padding-left:0',
                'width:100vw', 'min-width:100vw', 'max-width:100vw',
                'flex:1 1 100%', 'position:relative',
                'left:0', 'transform:none', 'overflow-x:hidden',
            ].join('!important;') + '!important;';
        }}
    }}

    /* ── Hook backdrop tap → close drawer ── */
    function hookBackdrop() {{
        var backdrop = d.getElementById('xtal-drawer-backdrop');
        if (backdrop && !backdrop._hooked) {{
            backdrop._hooked = true;
            backdrop.addEventListener('click', function() {{
                /* Find ✕ close button inside the sidebar element */
                var sidebar = d.querySelector('[data-testid="stSidebar"]');
                var closeBtn = null;
                if (sidebar) {{
                    var btns = sidebar.querySelectorAll('button');
                    for (var i = 0; i < btns.length; i++) {{
                        if (btns[i].textContent.trim() === '\u2715') {{
                            closeBtn = btns[i]; break;
                        }}
                    }}
                }}
                if (closeBtn) closeBtn.click();
            }});
        }}
    }}

    /* ── Run on load and after DOM settles ── */
    positionHamburger();
    applyMobileSidebar();
    hookBackdrop();
    setTimeout(function() {{ positionHamburger(); applyMobileSidebar(); hookBackdrop(); }}, 150);
    setTimeout(function() {{ positionHamburger(); applyMobileSidebar(); }}, 550);
}})();
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
    # ── Close button — uses .st-key-sidebar_close (Streamlit's auto-class) ──
    st.markdown("""
    <style>
    /* Hidden on desktop */
    .st-key-sidebar_close { display: none !important; }

    /* Mobile: fixed at top-right of the viewport, above the drawer */
    @media (max-width: 767px) {
        .st-key-sidebar_close {
            display: block !important;
            position: fixed !important;
            top: 14px !important;
            right: 14px !important;
            z-index: 1000001 !important;
            width: 44px !important;
            height: 44px !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .st-key-sidebar_close button {
            width: 44px !important;
            height: 44px !important;
            min-width: 44px !important;
            min-height: 44px !important;
            padding: 0 !important;
            background: rgba(35,35,35,0.95) !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            border-radius: 10px !important;
            color: #ccc !important;
            font-size: 18px !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: background 0.18s ease, color 0.18s ease !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.4) !important;
            -webkit-tap-highlight-color: transparent !important;
        }
        .st-key-sidebar_close button:hover,
        .st-key-sidebar_close button:active {
            background: rgba(60,60,60,0.98) !important;
            color: #fff !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    if st.button("✕", key="sidebar_close"):
        st.session_state.sidebar_open = False
        st.rerun()

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

st.markdown('<div class="created-by">Created by Devendra Saini</div>', unsafe_allow_html=True)

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
