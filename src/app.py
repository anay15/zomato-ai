"""
Phase 4: User Interface — Streamlit Dashboard
==============================================
AI-Powered Restaurant Recommendation System (Zomato Use Case)
"""

import sys
import json
import logging
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import ZomatoDataLoader
from src.retriever import RestaurantRetriever, RetrievalResult
from src.llm_connector import ZomatoLLMConnector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="ForkFinder — AI Restaurant Recommendations",
    page_icon="🍴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Cuisine visual map: (gradient-from, gradient-to, emoji) ──────────────────
CUISINE_VISUAL_MAP = {
    "north indian":    ("#3D0C0C", "#1A0505", "🍛"),
    "south indian":    ("#1A2E0C", "#0C1A05", "🥘"),
    "mughlai":         ("#2E1A0C", "#1A0C05", "🍖"),
    "biryani":         ("#2E220C", "#1A1405", "🍚"),
    "chinese":         ("#2E0C0C", "#180505", "🥡"),
    "italian":         ("#0C2E1A", "#051A0E", "🍝"),
    "pizza":           ("#2E1A0C", "#180E05", "🍕"),
    "burger":          ("#2E200C", "#1A1205", "🍔"),
    "mexican":         ("#2E1A0C", "#1A0E05", "🌮"),
    "japanese":        ("#0C1A2E", "#050E1A", "🍣"),
    "thai":            ("#0C2E20", "#051A12", "🍜"),
    "mediterranean":   ("#1A2E20", "#0C1A12", "🥙"),
    "american":        ("#2E1E0C", "#1A1205", "🍔"),
    "seafood":         ("#0C1E2E", "#05101A", "🦞"),
    "desserts":        ("#2E1A2E", "#1A0C1A", "🍰"),
    "bakery":          ("#2E200C", "#1A1205", "🥐"),
    "cafe":            ("#1E1A0C", "#120E05", "☕"),
    "continental":     ("#1A1A2E", "#0C0C1A", "🥗"),
    "fast food":       ("#2E1A0C", "#1A0C05", "🍟"),
    "street food":     ("#2E1E0C", "#1A1205", "🌯"),
    "arabian":         ("#2E1A0C", "#1A0C05", "🥙"),
    "andhra":          ("#2E0C0C", "#1A0505", "🌶️"),
    "default":         ("#091014", "#173538", "🍽️"),
}

def get_cuisine_visual(cuisines_str: str) -> tuple[str, str, str]:
    """Return (gradient-from, gradient-to, emoji) for the primary cuisine."""
    if not cuisines_str:
        return CUISINE_VISUAL_MAP["default"]
    first = cuisines_str.split(",")[0].strip().lower()
    for key, val in CUISINE_VISUAL_MAP.items():
        if key in first or first in key:
            return val
    return CUISINE_VISUAL_MAP["default"]


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

:root {
    --z-bg: #000000;
    --z-panel: #111111;
    --z-border: #252525;
    --z-text: #F8FAFC;
    --z-muted: #94A3B8;
    --z-red: #ef3348;
}

html, body, p, label, input, select, textarea, .stApp {
    font-family: 'Inter', sans-serif;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
    text-rendering: optimizeLegibility !important;
}

h1, h2, h3, h4, h5, h6, .hero-header h1, .results-title h2, .brand {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stApp { background: var(--z-bg); color: var(--z-text); }

.block-container { max-width: 1120px; padding: 1.35rem 2rem 3rem; }

#MainMenu, footer, header { visibility: hidden; }

[data-testid="stAppViewContainer"] > .main { border-left: 1px solid #151515; }
section[data-testid="stSidebar"] {
    background: #050505;
    border-right: 1px solid #171717;
    min-width: 280px !important;
    max-width: 280px !important;
}
section[data-testid="stSidebar"] > div:first-child { padding: 1.35rem 1rem 2rem; }
section[data-testid="stSidebar"] * { color: var(--z-text) !important; }


/* ── Streamlit Tabs → styled as nav ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #171717;
    margin-bottom: 2rem;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--z-muted) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.4rem !important;
    transition: color 200ms ease, border-color 200ms ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #fff !important; }
.stTabs [aria-selected="true"] {
    color: #fff !important;
    border-bottom: 2px solid var(--z-red) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Brand row above tabs ── */
.top-brand {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 0.65rem;
    margin-bottom: 0;
}
.brand { font-size: 1.05rem; font-weight: 800; color: #fff; }
.brand span { color: var(--z-red); }
.nav-icons { display: flex; gap: 1.5rem; color: var(--z-muted); font-size: 1rem; }

/* ── Hero Header ── */
.hero-header { text-align: center; padding: 0.2rem 1rem 2rem; }
.hero-header h1 { font-size: 2.6rem; font-weight: 800; color: #fff; margin: 0; line-height: 1.1; }
.hero-header .hero-icon { color: var(--z-red); margin-right: 0.65rem; }
.hero-header p { color: var(--z-muted); font-size: 1rem; margin: 0.95rem auto 0; max-width: 660px; line-height: 1.45; }

.section-divider { background: #171717; height: 1px; border: none; margin: 1.5rem 0; }

/* ── Stats ── */
.stats-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.4rem; margin: 0.7rem 0 2.2rem; }
.stat-card {
    background: var(--z-panel);
    border: 1px solid var(--z-border);
    border-radius: 8px;
    padding: 1.55rem 1.2rem;
    text-align: center;
    animation: fadeSlideUp 0.5s ease both;
}
.stat-card:nth-child(1) { animation-delay: 0.05s; }
.stat-card:nth-child(2) { animation-delay: 0.12s; }
.stat-card:nth-child(3) { animation-delay: 0.19s; }
.stat-card .stat-value { color: #fff; font-size: 2rem; font-weight: 800; }
.stat-card .stat-label { color: var(--z-muted); font-size: 0.63rem; letter-spacing: 0.18em; text-transform: uppercase; margin-top: 0.2rem; }

/* ── Status / Summary banners ── */
.status-banner, .summary-block {
    background: #151515;
    border: 1px solid var(--z-border);
    border-left: 4px solid var(--z-red);
    border-radius: 8px;
    color: #ead4d7;
    padding: 1.15rem 1.4rem;
    margin-bottom: 1.8rem;
    font-size: 0.9rem;
    line-height: 1.6;
    animation: fadeSlideUp 0.4s ease both;
}
.status-banner.relaxed { border-left-color: #37c871; color: #d0edda; }
.summary-block strong { color: #fff; display: block; margin-bottom: 0.35rem; }

/* ── Results title ── */
.results-title { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; margin: 1rem 0 1.8rem; }
.results-title h2 { color: #fff; font-size: 1.35rem; margin: 0; }
.results-title span { color: var(--z-muted); font-size: 0.76rem; font-style: italic; }

/* ── Recommendation Card ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

.rec-card {
    background: var(--z-panel);
    border: 1px solid var(--z-border);
    border-radius: 12px;
    margin-bottom: 1.55rem;
    overflow: hidden;
    position: relative;
    animation: fadeSlideUp 0.45s ease both;
    transition: border-color 220ms ease, transform 220ms ease, box-shadow 220ms ease;
}
.rec-card:hover {
    border-color: #3b3b3b;
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.55);
}
.rec-card:nth-child(1) { animation-delay: 0.05s; }
.rec-card:nth-child(2) { animation-delay: 0.12s; }
.rec-card:nth-child(3) { animation-delay: 0.19s; }
.rec-card:nth-child(4) { animation-delay: 0.26s; }
.rec-card:nth-child(5) { animation-delay: 0.33s; }

.rec-feature { display: grid; grid-template-columns: minmax(230px, 34%) 1fr; }

.food-visual {
    min-height: 260px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}
.food-emoji {
    font-size: 5rem;
    position: relative;
    z-index: 1;
    animation: floatEmoji 4s ease-in-out infinite;
    filter: drop-shadow(0 4px 16px rgba(0,0,0,0.6));
    user-select: none;
}
@keyframes floatEmoji {
    0%, 100% { transform: translateY(0px) rotate(-3deg); }
    50%       { transform: translateY(-8px) rotate(3deg); }
}
.food-visual::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, rgba(0,0,0,0.08), rgba(0,0,0,0.52));
    z-index: 0;
}

/* Favorite heart button on card */
.fav-btn-overlay {
    position: absolute;
    top: 0.75rem;
    right: 0.75rem;
    z-index: 2;
    background: rgba(0,0,0,0.5);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 50%;
    width: 2rem;
    height: 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 1rem;
    transition: background 200ms ease, transform 200ms ease;
    backdrop-filter: blur(4px);
}
.fav-btn-overlay:hover { background: rgba(239,51,72,0.3); transform: scale(1.15); }
.fav-btn-overlay.saved { background: rgba(239,51,72,0.5); }

.rec-body { padding: 1.85rem 1.9rem 1.7rem; }
.rank-badge {
    width: 3.05rem; height: 2.45rem; border-radius: 4px;
    font-size: 1.2rem; font-weight: 800;
    display: inline-flex; align-items: center; justify-content: center;
    background: var(--z-red); color: #fff;
    border: 1px solid rgba(255,255,255,0.12);
    position: absolute; left: 1rem; top: 1rem; z-index: 2;
}
.rank-2, .rank-3, .rank-other { background: #202727; color: #fff; }

.card-name-row { display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 0.65rem; }
.card-name { font-size: 1.75rem; font-weight: 800; color: #fff; line-height: 1.15; }
.card-locality { color: var(--z-muted); font-size: 0.85rem; margin-top: 0.45rem; }

.rating-badge {
    display: inline-flex; align-items: center; gap: 0.25rem;
    padding: 0.55rem 0.75rem;
    background: #191919; color: #fff;
    border: 1px solid #363636;
    font-size: 0.82rem; font-weight: 700; white-space: nowrap;
}

.chips-row { display: flex; flex-wrap: wrap; gap: 0.55rem; margin: 1.05rem 0; }
.chip {
    padding: 0.2rem 0.65rem; border-radius: 3px;
    background: #241c1d; color: #f2d9dc;
    border: 1px solid #3a2c2f;
    font-size: 0.72rem; font-weight: 700;
}
.cost-pill {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.55rem 0.75rem; border-radius: 3px;
    background: #1c1718; color: #f0d8db;
    border: 1px solid #302628; font-size: 0.75rem; font-weight: 600;
}

.ai-reasoning {
    background: #070707; border-left: 3px solid var(--z-red);
    padding: 0.8rem 1rem; margin-top: 1.35rem;
    color: #e5cfd2; font-size: 0.88rem; line-height: 1.65;
}
.ai-label {
    color: var(--z-red); font-size: 0.66rem;
    font-style: normal; font-weight: 800;
    letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 0.6rem;
}

/* ── Favorites / History tab content ── */
.empty-state {
    text-align: center; padding: 5rem 2rem;
    color: rgba(255,255,255,0.2);
    animation: fadeSlideUp 0.4s ease both;
}
.empty-state .empty-icon { font-size: 4rem; margin-bottom: 1rem; }
.empty-state p { font-size: 1rem; font-weight: 500; }

.history-item {
    background: var(--z-panel);
    border: 1px solid var(--z-border);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    display: flex; align-items: center; justify-content: space-between;
    animation: fadeSlideUp 0.4s ease both;
    transition: border-color 200ms ease;
}
.history-item:hover { border-color: #3b3b3b; }
.history-meta { color: var(--z-muted); font-size: 0.75rem; margin-top: 0.3rem; }

/* ── Skeleton Loader ── */
@keyframes skeleton-shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position:  400px 0; }
}
.skeleton-card {
    background: var(--z-panel); border: 1px solid var(--z-border);
    border-radius: 8px; padding: 1.5rem; margin-bottom: 1.2rem; overflow: hidden;
}
.skeleton-line {
    height: 14px; border-radius: 8px; margin-bottom: 0.75rem;
    background: linear-gradient(90deg,
        rgba(255,255,255,0.04) 0%,
        rgba(255,255,255,0.12) 50%,
        rgba(255,255,255,0.04) 100%);
    background-size: 800px 100%;
    animation: skeleton-shimmer 1.4s ease-in-out infinite;
}
.skeleton-line.title    { height: 22px; width: 55%; }
.skeleton-line.subtitle { width: 35%; }
.skeleton-line.short    { width: 25%; }
.skeleton-line.reasoning { height: 60px; width: 100%; margin-top: 1rem; }
.skeleton-spinner-row {
    display: flex; align-items: center; gap: 0.75rem;
    color: var(--z-muted); font-size: 0.9rem; margin-bottom: 1.2rem;
}
.skeleton-spinner {
    width: 18px; height: 18px;
    border: 2px solid rgba(255,255,255,0.15);
    border-top-color: var(--z-red);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Favorite/unfavorite button (Streamlit button styled) ── */
div[data-testid="column"] button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #252525 !important;
    color: var(--z-muted) !important;
    font-size: 0.78rem !important;
    padding: 0.3rem 0.8rem !important;
    border-radius: 4px !important;
    transition: border-color 150ms ease, color 150ms ease !important;
}
div[data-testid="column"] button[kind="secondary"]:hover {
    border-color: var(--z-red) !important;
    color: var(--z-red) !important;
}

@media (max-width: 900px) {
    .rec-feature { grid-template-columns: 1fr; }
    .food-visual { min-height: 190px; }
    .stats-row { grid-template-columns: 1fr; }
    .hero-header h1 { font-size: 2rem; }
}
</style>
""", unsafe_allow_html=True)

# ── Hide the sidebar collapse button so it always stays open ─────────────────
st.markdown("""
<style>
/* Hide every variant of Streamlit's sidebar collapse/expand toggle */
button[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] button[kind="header"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# ── Cached Loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading restaurant database...")
def load_retriever():
    return RestaurantRetriever(workspace_dir=str(ROOT))

@st.cache_resource(show_spinner=False)
def load_connector():
    return ZomatoLLMConnector()

@st.cache_data(show_spinner=False)
def get_unique_locations(_df: pd.DataFrame):
    locs = set(_df['locality'].dropna().unique()) | set(_df['city'].dropna().unique())
    return sorted(locs - {"Unknown Locality", "Unknown City"})

@st.cache_data(show_spinner=False)
def get_unique_cuisines(_df: pd.DataFrame):
    all_cuisines = set()
    for entry in _df['cuisines'].dropna():
        for c in entry.split(","):
            c = c.strip()
            if c and c.lower() != "multi-cuisine":
                all_cuisines.add(c)
    return sorted(all_cuisines)


# ── Helpers ───────────────────────────────────────────────────────────────────
def provider_display_name(provider: str) -> str:
    return {"groq": "Groq AI", "gemini": "Gemini AI", "openai": "OpenAI"}.get(provider.lower(), "AI")

def rating_class(rating: float) -> str:
    if rating >= 4.5: return "rating-high"
    elif rating >= 3.5: return "rating-mid"
    return "rating-low"

def rank_class(rank: int) -> str:
    return {1: "rank-1", 2: "rank-2", 3: "rank-3"}.get(rank, "rank-other")

def budget_label(cost: float) -> str:
    if cost < 400: return "Budget-Friendly"
    elif cost < 1000: return "Moderate"
    return "Premium"


def build_skeleton_loader_html(count: int = 3, message: str = "Generating recommendations...") -> str:
    cards = "".join(
        '<div class="skeleton-card">'
        '<div class="skeleton-line title"></div>'
        '<div class="skeleton-line subtitle"></div>'
        '<div class="skeleton-line short"></div>'
        '<div class="skeleton-line reasoning"></div>'
        '</div>'
        for _ in range(count)
    )
    return (
        '<div class="skeleton-spinner-row">'
        '<div class="skeleton-spinner"></div>'
        f'<span>{message}</span>'
        '</div>'
        + cards
    )


def build_relaxation_banner_html(result: RetrievalResult) -> str:
    if not result.user_message:
        return ""
    css_class = "relaxed" if result.status != "global_fallback" else ""
    nearby_hint = ""
    if result.nearby_localities:
        chips = "".join(f'<span class="chip">{loc}</span>' for loc in result.nearby_localities[:5])
        nearby_hint = f'<div class="chips-row" style="margin-top:0.6rem;">{chips}</div>'
    return f'<div class="status-banner {css_class}">💡 {result.user_message}{nearby_hint}</div>'


def build_card_html(rec: dict, rank: int, is_saved: bool = False) -> str:
    r_class = rank_class(rank)
    rating = rec.get("aggregate_rating", 0)
    r_badge_class = rating_class(rating)
    cost = rec.get("average_cost_for_two", 0)
    cuisines_str = rec.get("cuisines", "")
    cuisines = [c.strip() for c in cuisines_str.split(",")][:4]
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in cuisines if c)
    grad_from, grad_to, emoji = get_cuisine_visual(cuisines_str)
    heart = "❤️" if is_saved else "🤍"

    return f"""
    <div class="rec-card rec-feature">
        <div class="food-visual" style="background: radial-gradient(circle at 50% 60%, {grad_from}CC, {grad_to} 70%);">
            <span class="rank-badge {r_class}">#{rank}</span>
            <span class="food-emoji">{emoji}</span>
            <span class="fav-btn-overlay {'saved' if is_saved else ''}" title="{'Saved' if is_saved else 'Save to favorites'}">{heart}</span>
        </div>
        <div class="rec-body">
            <div class="card-name-row">
                <div>
                    <div class="card-name">{rec.get('name', 'Restaurant')}</div>
                    <div class="card-locality">📍 {rec.get('locality', '')}, {rec.get('city', '')}</div>
                </div>
                <div style="margin-left:auto; display:flex; flex-direction:column; align-items:flex-end; gap:0.4rem;">
                    <span class="rating-badge {r_badge_class}">{rating:.1f} <span style="font-size:0.72rem;">/5.0</span></span>
                    <span style="font-size:0.66rem; color:#d4bdc0; font-weight:800; letter-spacing:0.08em;">{rec.get('votes', 0):,} REVIEWS</span>
                </div>
            </div>
            <div class="chips-row">{chips_html}</div>
            <div style="display:flex; gap:0.6rem; align-items:center; flex-wrap:wrap; margin-top:0.2rem;">
                <span class="cost-pill">₹{int(cost):,} for two</span>
                <span class="cost-pill">{budget_label(cost)}</span>
            </div>
            <div class="ai-reasoning">
                <div class="ai-label">AI Insight</div>
                {rec.get('reasoning', '')}
            </div>
        </div>
    </div>
    """


def render_card_with_fav_btn(rec: dict, rank: int, context: str = "discover"):
    """Render card HTML then a Streamlit save/unsave button below it."""
    name = rec.get("name", "")
    fav_ids = {r.get("name") for r in st.session_state.favorites}
    is_saved = name in fav_ids

    st.markdown(build_card_html(rec, rank, is_saved=is_saved), unsafe_allow_html=True)

    btn_label = "❤️ Remove from Favorites" if is_saved else "🤍 Save to Favorites"
    key = f"fav_{context}_{rank}_{name}"
    col1, col2 = st.columns([5, 2])
    with col2:
        if st.button(btn_label, key=key):
            if is_saved:
                st.session_state.favorites = [r for r in st.session_state.favorites if r.get("name") != name]
            else:
                st.session_state.favorites.append(rec)
            # No st.rerun() — Streamlit button clicks already trigger a rerun automatically.


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    connector = load_connector()
    ai_label = provider_display_name(connector.provider)

    # ── Session state init ────────────────────────────────────────────────────
    if "search_data" not in st.session_state:
        st.session_state.search_data = {
            "location": "", "cuisines": [], "budget": "Any",
            "min_rating": 4.0, "user_query": "", "search_clicked": False
        }
    if "favorites" not in st.session_state:
        st.session_state.favorites = []
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    # last_results persists across reruns so fav-button reruns don't wipe them
    if "last_results" not in st.session_state:
        st.session_state.last_results = None
    # Track the last nonce we already ran — any rerun returning the same nonce
    # is a stale cached component value, not a fresh click
    if "last_search_nonce" not in st.session_state:
        st.session_state.last_search_nonce = 0

    # Load resources
    try:
        retriever = load_retriever()
        df = retriever.df
    except Exception as e:
        st.error(f"Failed to initialize the recommendation engine: {e}")
        st.stop()

    locations = get_unique_locations(df)
    cuisines_list = get_unique_cuisines(df)

    # ── Brand row ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="top-brand">
        <div class="brand">Fork<span>Finder</span></div>
        <div class="nav-icons"><span>&#9825;</span><span>&#9678;</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_discover, tab_favorites, tab_history = st.tabs(["🔍 Discover", "❤️ Favorites", "🕐 History"])

    # ── Sidebar ───────────────────────────────────────────────────────────────
    custom_sidebar = components.declare_component(
        "zomato_sidebar",
        path=str(ROOT / "src" / "sidebar")
    )
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 0.5rem;">
            <div style="font-size:1.35rem; color:#ef3348; font-family:'Plus Jakarta Sans',sans-serif;">⚙️</div>
            <div style="font-size:1.35rem; font-weight:800; color:#fff; margin-top:0.55rem; font-family:'Plus Jakarta Sans',sans-serif;">Search Preferences</div>
            <div style="font-size:0.75rem; color:#94A3B8; margin-top:0.2rem;">Personalise your palate</div>
        </div>
        <hr style="border-color:#171717; margin: 1.35rem 0;">
        """, unsafe_allow_html=True)

        result = custom_sidebar(
            locations=locations,
            cuisines=cuisines_list,
            default_values=st.session_state.search_data,
            key="zomato_sidebar_stable",   # explicit key prevents iframe remount when widget tree changes
        )
        if result:
            st.session_state.search_data = result

    search_data = st.session_state.search_data
    location = search_data.get("location", "")
    selected_cuisines = search_data.get("cuisines", [])
    budget = search_data.get("budget", "Any")
    min_rating = float(search_data.get("min_rating", 4.0))
    user_query = search_data.get("user_query", "")

    # A click is "fresh" only when the component reports search_clicked=True
    # AND the nonce is higher than the last one we already processed.
    incoming_nonce = int(search_data.get("search_nonce", 0))
    is_fresh_click = (
        search_data.get("search_clicked", False)
        and incoming_nonce > st.session_state.last_search_nonce
    )
    logger.info(
        "RERUN | nonce=%s last_nonce=%s search_clicked=%s is_fresh=%s location=%s",
        incoming_nonce,
        st.session_state.last_search_nonce,
        search_data.get("search_clicked"),
        is_fresh_click,
        location,
    )
    if is_fresh_click:
        st.session_state.last_search_nonce = incoming_nonce

    search_clicked = is_fresh_click

    # ─────────────────────────────────────────────────────────────────────────
    # TAB: DISCOVER
    # ─────────────────────────────────────────────────────────────────────────
    with tab_discover:
        avg_rating = df['aggregate_rating'].mean()
        n_cuisines = len(cuisines_list)
        st.markdown(f"""
        <div class="hero-header">
            <h1><span class="hero-icon">🍴</span>ForkFinder</h1>
            <p>Tell us what you're craving — ForkFinder uses {ai_label} &amp; real Zomato data to find your perfect table.</p>
        </div>
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value">{len(df):,}</div>
                <div class="stat-label">Restaurants</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{n_cuisines}</div>
                <div class="stat-label">Cuisine Types</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_rating:.1f} <span style="color:#ef3348;">★</span></div>
                <div class="stat-label">Avg Dataset Rating</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Run a fresh search when triggered ─────────────────────────────────
        if search_clicked:
            if not location:
                st.warning("⚠️ Please select a location from the sidebar.")
            else:
                effective_query = user_query.strip() or f"Top-rated restaurant in {location}"
                budget_tier = "" if budget == "Any" else budget

                with st.spinner("🔍 Searching and ranking the best restaurants..."):
                    retrieval = retriever.retrieve_candidates(
                        location=location,
                        cuisines=selected_cuisines if selected_cuisines else None,
                        budget_tier=budget_tier,
                        min_rating=min_rating,
                        max_candidates=15,
                    )
                    candidates = RestaurantRetriever.dataframe_to_records(retrieval.candidates)

                if not candidates:
                    st.error("No restaurants found even after relaxing constraints. Try a different location or cuisine.")
                else:
                    llm_message = f"{ai_label} is crafting personalised recommendations..."
                    skeleton_slot = st.empty()
                    skeleton_slot.markdown(build_skeleton_loader_html(count=3, message=llm_message), unsafe_allow_html=True)
                    try:
                        llm_result = connector.generate_recommendations(candidates=candidates, user_query=effective_query)
                    finally:
                        skeleton_slot.empty()

                    recommendations = llm_result.get("recommendations", [])
                    summary = llm_result.get("summary", "")
                    banner_html = build_relaxation_banner_html(retrieval)

                    # Persist results — survives the 150ms reset rerun and fav-button reruns
                    st.session_state.last_results = {
                        "recommendations": recommendations,
                        "summary": summary,
                        "banner_html": banner_html,
                        "location": location,
                        "candidates_count": len(candidates),
                        "query": effective_query,
                        "location_for_history": location,
                        "cuisines": selected_cuisines,
                        "budget": budget,
                        "min_rating": min_rating,
                    }

                    # Add to history (only on fresh search)
                    history_entry = {
                        "location": location,
                        "cuisines": selected_cuisines,
                        "budget": budget,
                        "min_rating": min_rating,
                        "query": effective_query,
                        "result_count": len(recommendations),
                        "recommendations": recommendations,
                    }
                    if not st.session_state.search_history or st.session_state.search_history[-1].get("query") != effective_query:
                        st.session_state.search_history.append(history_entry)

        # ── Render stored results (every rerun, not just on search_clicked) ───
        stored = st.session_state.last_results
        if stored:
            recommendations = stored["recommendations"]
            summary = stored["summary"]
            banner_html = stored.get("banner_html", "")
            loc_label = stored["location"]
            candidates_count = stored["candidates_count"]

            if banner_html:
                st.markdown(banner_html, unsafe_allow_html=True)

            if "offline" in summary.lower() or "_API_KEY" in summary:
                st.markdown("""
                <div class="status-banner">
                    🔌 <strong>Offline Mode:</strong> Results ranked by community ratings.
                    Add your LLM API key to <code>.env</code> for AI-personalised recommendations.
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="results-title">
                <h2>🔥 Top {len(recommendations)} Picks for "{loc_label}"</h2>
                <span>Filtered from {candidates_count} candidates</span>
            </div>
            """, unsafe_allow_html=True)

            if summary:
                st.markdown(f'<div class="summary-block"><strong>AI Summary</strong>{summary}</div>', unsafe_allow_html=True)

            for rec in recommendations:
                render_card_with_fav_btn(rec, rec.get("rank", 1), context="discover")

            st.markdown(f"""
            <hr class="section-divider">
            <div style="text-align:center; color:rgba(255,255,255,0.2); font-size:0.75rem; padding-bottom:2rem;">
                ForkFinder • Powered by {ai_label} • Built with Streamlit
            </div>
            """, unsafe_allow_html=True)

        elif not search_clicked:
            # No results yet and no active search — show the landing prompt
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🍽️</div>
                <p>Set your preferences in the sidebar and click <strong style="color:rgba(255,255,255,0.4)">Find Restaurants</strong></p>
                <div style="font-size:0.85rem; margin-top:0.5rem;">
                    Our AI will analyse thousands of restaurants and hand-pick the best matches for you
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB: FAVORITES
    # ─────────────────────────────────────────────────────────────────────────
    with tab_favorites:
        favs = st.session_state.favorites
        if not favs:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🤍</div>
                <p>No favorites saved yet</p>
                <div style="font-size:0.85rem; margin-top:0.5rem;">
                    Hit <strong style="color:rgba(255,255,255,0.4)">Save to Favorites</strong>
                    on any recommendation in the Discover tab
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="results-title">
                <h2>❤️ Your Favorites ({len(favs)})</h2>
            </div>
            """, unsafe_allow_html=True)
            for i, rec in enumerate(favs, 1):
                render_card_with_fav_btn(rec, i, context="favorites")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB: HISTORY
    # ─────────────────────────────────────────────────────────────────────────
    with tab_history:
        history = st.session_state.search_history
        if not history:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🕐</div>
                <p>No search history yet</p>
                <div style="font-size:0.85rem; margin-top:0.5rem;">
                    Your past searches will appear here
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="results-title">
                <h2>🕐 Search History ({len(history)})</h2>
            </div>
            """, unsafe_allow_html=True)

            for i, entry in enumerate(reversed(history)):
                idx = len(history) - i
                cuisine_str = ", ".join(entry.get("cuisines", [])) or "Any cuisine"
                st.markdown(f"""
                <div class="history-item">
                    <div>
                        <div style="font-weight:700; color:#fff; font-size:0.92rem;">
                            📍 {entry.get('location', '')}
                        </div>
                        <div class="history-meta">
                            {cuisine_str} &nbsp;·&nbsp;
                            Budget: {entry.get('budget','Any')} &nbsp;·&nbsp;
                            Min rating: {entry.get('min_rating', 4.0):.1f}★ &nbsp;·&nbsp;
                            {entry.get('result_count', 0)} results
                        </div>
                        <div style="color:#64748B; font-size:0.72rem; margin-top:0.2rem; font-style:italic;">
                            "{entry.get('query','')}"
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"View {entry.get('result_count', 0)} results from this search", expanded=False):
                    recs = entry.get("recommendations", [])
                    if recs:
                        for rec in recs:
                            render_card_with_fav_btn(rec, rec.get("rank", 1), context=f"history_{idx}")
                    else:
                        st.markdown('<div style="color:#64748B; padding:1rem;">No results stored.</div>', unsafe_allow_html=True)

                if st.button("🔄 Re-run this search", key=f"rerun_{idx}"):
                    st.session_state.search_data.update({
                        "location": entry.get("location", ""),
                        "cuisines": entry.get("cuisines", []),
                        "budget": entry.get("budget", "Any"),
                        "min_rating": entry.get("min_rating", 4.0),
                        "user_query": entry.get("query", ""),
                        "search_clicked": True,
                    })
                    st.rerun()


if __name__ == "__main__":
    main()
