"""
Phase 4: User Interface — Streamlit Dashboard
==============================================
AI-Powered Restaurant Recommendation System (Zomato Use Case)

Features:
- Wide layout with custom sidebar for preference inputs
- Dynamically loaded locations & cuisines from the Zomato dataset
- Spinner while LLM generates recommendations
- Premium styled recommendation cards with rating badges, cost info & AI reasoning
- Fallback awareness banner when offline mode is active
"""

import sys
import json
import logging
import streamlit as st
import pandas as pd
from pathlib import Path

# ── Ensure project root is on sys.path ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import ZomatoDataLoader
from src.retriever import RestaurantRetriever, RetrievalResult
from src.llm_connector import ZomatoLLMConnector

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zomato AI | Smart Restaurant Recommendations",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium CSS Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── App Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] * { color: #e8e8f0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stTextArea label { 
    color: #a0a0c0 !important;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Main Header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcb77, #4d96ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}
.hero-header p {
    color: rgba(255,255,255,0.55);
    font-size: 1rem;
    margin-top: 0.5rem;
}

/* ── Section Divider ── */
.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
    margin: 1.5rem 0;
}

/* ── Status Banner ── */
.status-banner {
    background: rgba(255, 193, 7, 0.10);
    border: 1px solid rgba(255, 193, 7, 0.30);
    border-radius: 10px;
    padding: 0.75rem 1.2rem;
    color: #ffd93d;
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.status-banner.relaxed {
    background: rgba(107, 203, 119, 0.10);
    border-color: rgba(107, 203, 119, 0.30);
    color: #6bcb77;
}

/* ── Recommendation Card ── */
.rec-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.3s ease, transform 0.2s ease;
    position: relative;
    overflow: hidden;
}
.rec-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #ff6b6b, #ffd93d, #4d96ff);
    opacity: 0;
    transition: opacity 0.3s;
}
.rec-card:hover::before { opacity: 1; }
.rec-card:hover {
    border-color: rgba(255,255,255,0.2);
    transform: translateY(-2px);
}

/* ── Card Rank Badge ── */
.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2rem; height: 2rem;
    border-radius: 50%;
    font-size: 0.85rem;
    font-weight: 700;
    margin-right: 0.8rem;
    flex-shrink: 0;
}
.rank-1 { background: linear-gradient(135deg, #ffd700, #ffa500); color: #1a1a2e; }
.rank-2 { background: linear-gradient(135deg, #c0c0c0, #a8a8a8); color: #1a1a2e; }
.rank-3 { background: linear-gradient(135deg, #cd7f32, #b8651a); color: #fff; }
.rank-other { background: rgba(255,255,255,0.1); color: rgba(255,255,255,0.6); }

/* ── Card Name Row ── */
.card-name-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.8rem;
}
.card-name {
    font-size: 1.15rem;
    font-weight: 700;
    color: #ffffff;
    flex: 1;
}
.card-locality {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.45);
    margin-top: 0.1rem;
}

/* ── Rating Badge ── */
.rating-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.65rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
    white-space: nowrap;
}
.rating-high { background: rgba(107,203,119,0.18); color: #6bcb77; border: 1px solid rgba(107,203,119,0.35); }
.rating-mid  { background: rgba(255,193,7,0.18);   color: #ffd93d; border: 1px solid rgba(255,193,7,0.35); }
.rating-low  { background: rgba(255,107,107,0.18); color: #ff6b6b; border: 1px solid rgba(255,107,107,0.35); }

/* ── Chips ── */
.chips-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.7rem 0; }
.chip {
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    background: rgba(77,150,255,0.12);
    color: #93c5fd;
    border: 1px solid rgba(77,150,255,0.25);
}

/* ── Cost Pill ── */
.cost-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(255,255,255,0.07);
    color: rgba(255,255,255,0.65);
    border: 1px solid rgba(255,255,255,0.1);
}

/* ── AI Reasoning Block ── */
.ai-reasoning {
    background: rgba(77,150,255,0.07);
    border-left: 3px solid #4d96ff;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    margin-top: 0.9rem;
    color: rgba(255,255,255,0.78);
    font-size: 0.88rem;
    line-height: 1.65;
    font-style: italic;
}

/* ── Summary Block ── */
.summary-block {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 1.8rem;
    color: rgba(255,255,255,0.6);
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ── Search Button ── */
.stButton > button {
    width: 100%;
    padding: 0.65rem 1rem;
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    font-size: 0.95rem;
    cursor: pointer;
    transition: opacity 0.2s;
    letter-spacing: 0.02em;
}
.stButton > button:hover { opacity: 0.9; }

/* ── Metric Cards ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-card .stat-value {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff6b6b, #4d96ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-card .stat-label {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 0.2rem;
}

/* ── Input overrides ── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: white !important;
}
.stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: rgba(255,255,255,0.9) !important;
}

/* ── Skeleton Loader ── */
@keyframes skeleton-shimmer {
    0% { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
.skeleton-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    overflow: hidden;
}
.skeleton-line {
    height: 14px;
    border-radius: 8px;
    margin-bottom: 0.75rem;
    background: linear-gradient(
        90deg,
        rgba(255,255,255,0.04) 0%,
        rgba(255,255,255,0.12) 50%,
        rgba(255,255,255,0.04) 100%
    );
    background-size: 800px 100%;
    animation: skeleton-shimmer 1.4s ease-in-out infinite;
}
.skeleton-line.title { height: 22px; width: 55%; }
.skeleton-line.subtitle { width: 35%; }
.skeleton-line.short { width: 25%; }
.skeleton-line.reasoning { height: 60px; width: 100%; margin-top: 1rem; }
.skeleton-spinner-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    color: rgba(255,255,255,0.55);
    font-size: 0.9rem;
    margin-bottom: 1.2rem;
}
.skeleton-spinner {
    width: 18px;
    height: 18px;
    border: 2px solid rgba(255,255,255,0.15);
    border-top-color: #ff6b6b;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
</style>
""", unsafe_allow_html=True)


# ── Cached Resource Loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading restaurant database...")
def load_retriever():
    return RestaurantRetriever(workspace_dir=str(ROOT))

@st.cache_resource(show_spinner=False)
def load_connector():
    return ZomatoLLMConnector()

@st.cache_data(show_spinner=False)
def get_unique_locations(_df: pd.DataFrame):
    """Returns sorted list of unique localities + cities."""
    locs = set(_df['locality'].dropna().unique()) | set(_df['city'].dropna().unique())
    return sorted(locs - {"Unknown Locality", "Unknown City"})

@st.cache_data(show_spinner=False)
def get_unique_cuisines(_df: pd.DataFrame):
    """Expands comma-separated cuisine strings into a sorted unique list."""
    all_cuisines = set()
    for entry in _df['cuisines'].dropna():
        for c in entry.split(","):
            c = c.strip()
            if c and c.lower() != "multi-cuisine":
                all_cuisines.add(c)
    return sorted(all_cuisines)


# ── Rating Badge Helper ───────────────────────────────────────────────────────
def provider_display_name(provider: str) -> str:
    return {
        "groq": "Groq AI",
        "gemini": "Gemini AI",
        "openai": "OpenAI",
    }.get(provider.lower(), "AI")


def rating_class(rating: float) -> str:
    if rating >= 4.5:
        return "rating-high"
    elif rating >= 3.5:
        return "rating-mid"
    return "rating-low"

def rank_class(rank: int) -> str:
    return {1: "rank-1", 2: "rank-2", 3: "rank-3"}.get(rank, "rank-other")

def budget_label(cost: float) -> str:
    if cost < 400:
        return "Budget-Friendly"
    elif cost < 1000:
        return "Moderate"
    return "Premium"


def filter_locations(locations: list[str], query: str) -> list[str]:
    """Narrows location options for sidebar autocomplete-style search."""
    if not query.strip():
        return locations
    q = query.strip().lower()
    return [loc for loc in locations if q in loc.lower()]


def build_skeleton_loader_html(count: int = 3, message: str = "Generating recommendations...") -> str:
    cards = []
    for _ in range(count):
        cards.append("""
        <div class="skeleton-card">
            <div class="skeleton-line title"></div>
            <div class="skeleton-line subtitle"></div>
            <div class="skeleton-line short"></div>
            <div class="skeleton-line reasoning"></div>
        </div>
        """)
    return f"""
    <div class="skeleton-spinner-row">
        <div class="skeleton-spinner"></div>
        <span>{message}</span>
    </div>
    {''.join(cards)}
    """


def build_relaxation_banner_html(result: RetrievalResult) -> str:
    """Render Phase 5 fallback notification for the UI."""
    if not result.user_message:
        return ""
    css_class = "relaxed" if result.status != "global_fallback" else ""
    nearby_hint = ""
    if result.nearby_localities:
        chips = "".join(
            f'<span class="chip">{loc}</span>' for loc in result.nearby_localities[:5]
        )
        nearby_hint = f'<div class="chips-row" style="margin-top:0.6rem;">{chips}</div>'
    return f"""
    <div class="status-banner {css_class}">
        💡 {result.user_message}
        {nearby_hint}
    </div>
    """


def build_card_html(rec: dict, rank: int) -> str:
    r_class = rank_class(rank)
    rating = rec.get("aggregate_rating", 0)
    r_badge_class = rating_class(rating)
    cost = rec.get("average_cost_for_two", 0)
    cuisines = [c.strip() for c in rec.get("cuisines", "").split(",")][:4]
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in cuisines if c)

    return f"""
    <div class="rec-card">
        <div class="card-name-row">
            <span class="rank-badge {r_class}">#{rank}</span>
            <div>
                <div class="card-name">{rec.get('name', 'Restaurant')}</div>
                <div class="card-locality">📍 {rec.get('locality', '')}, {rec.get('city', '')}</div>
            </div>
            <div style="margin-left:auto; display:flex; flex-direction:column; align-items:flex-end; gap:0.4rem;">
                <span class="rating-badge {r_badge_class}">⭐ {rating:.1f} / 5.0</span>
                <span style="font-size:0.72rem; color:rgba(255,255,255,0.35);">{rec.get('votes', 0):,} reviews</span>
            </div>
        </div>
        <div class="chips-row">{chips_html}</div>
        <div style="display:flex; gap:0.6rem; align-items:center; flex-wrap:wrap; margin-top:0.2rem;">
            <span class="cost-pill">💰 ₹{int(cost):,} for two</span>
            <span class="cost-pill">🏷️ {budget_label(cost)}</span>
        </div>
        <div class="ai-reasoning">
            🤖 <strong>AI Insight:</strong> {rec.get('reasoning', '')}
        </div>
    </div>
    """


# ── Recommendation Card Renderer ──────────────────────────────────────────────
def render_card(rec: dict, rank: int):
    st.markdown(build_card_html(rec, rank), unsafe_allow_html=True)


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    # Hero Header
    connector = load_connector()
    ai_label = provider_display_name(connector.provider)

    st.markdown(f"""
    <div class="hero-header">
        <h1>🍽️ Zomato AI Recommender</h1>
        <p>Personalised restaurant recommendations powered by {ai_label} & real Zomato data</p>
    </div>
    <hr class="section-divider">
    """, unsafe_allow_html=True)

    # Load resources
    try:
        retriever = load_retriever()
        df = retriever.df
    except Exception as e:
        st.error(f"Failed to initialize the recommendation engine: {e}")
        st.stop()

    locations = get_unique_locations(df)
    cuisines_list = get_unique_cuisines(df)

    # ── Dataset Stats ─────────────────────────────────────────────────────────
    avg_rating = df['aggregate_rating'].mean()
    n_cuisines = len(cuisines_list)
    st.markdown(f"""
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
            <div class="stat-value">{avg_rating:.1f}⭐</div>
            <div class="stat-label">Avg Dataset Rating</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar: Search Controls ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 0.5rem;">
            <div style="font-size:1.8rem;">🔍</div>
            <div style="font-size:1rem; font-weight:700; color:#fff; margin-top:0.3rem;">Search Preferences</div>
            <div style="font-size:0.75rem; color:rgba(255,255,255,0.35); margin-top:0.2rem;">
                Tell us what you're looking for
            </div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.08); margin: 0.8rem 0;">
        """, unsafe_allow_html=True)

        # Location search + selector (autocomplete-style filter)
        location_search = st.text_input(
            "🔎 Search location",
            placeholder="Type to filter neighbourhoods...",
            key="location_search",
        )
        filtered_locations = filter_locations(locations, location_search)
        location = st.selectbox(
            "📍 Location",
            options=[""] + filtered_locations,
            format_func=lambda x: "Select a neighbourhood..." if x == "" else x,
            key="location_select",
        )

        # Cuisines (multi-select)
        selected_cuisines = st.multiselect(
            "🍜 Cuisines",
            options=cuisines_list,
            placeholder="Any cuisine...",
            key="cuisine_select"
        )

        # Budget
        budget = st.radio(
            "💰 Budget",
            options=["Any", "Low", "Medium", "High"],
            horizontal=True,
            key="budget_radio"
        )

        # Min Rating
        min_rating = st.slider(
            "⭐ Minimum Rating",
            min_value=2.0, max_value=5.0,
            value=4.0, step=0.1,
            key="rating_slider"
        )

        st.markdown("<hr style='border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

        # Qualitative Search
        user_query = st.text_area(
            "✨ What are you looking for?",
            placeholder="e.g., A cozy rooftop place for a date night with great cocktails...",
            height=110,
            key="query_text"
        )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        search_clicked = st.button("🚀 Find Restaurants", key="search_btn")

    # ── Main Panel: Results ────────────────────────────────────────────────────
    if not search_clicked:
        # Placeholder when no search yet
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem; color:rgba(255,255,255,0.25);">
            <div style="font-size:4rem; margin-bottom:1rem;">🍽️</div>
            <div style="font-size:1.1rem; font-weight:500;">
                Set your preferences in the sidebar and click <strong style="color:rgba(255,255,255,0.4)">Find Restaurants</strong>
            </div>
            <div style="font-size:0.85rem; margin-top:0.5rem;">
                Our AI will analyse thousands of restaurants and hand-pick the best matches for you
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Validate input
    if not location:
        st.warning("⚠️ Please select a location from the sidebar to search.")
        return

    effective_query = user_query.strip() or f"Top-rated restaurant in {location}"
    budget_tier = "" if budget == "Any" else budget

    # ── Run Retrieval + LLM ────────────────────────────────────────────────────
    with st.spinner("🔍 Searching and ranking the best restaurants for you..."):
        retrieval = retriever.retrieve_candidates(
            location=location,
            cuisines=selected_cuisines if selected_cuisines else None,
            budget_tier=budget_tier,
            min_rating=min_rating,
            max_candidates=15,
        )
        candidates = RestaurantRetriever.dataframe_to_records(retrieval.candidates)

    banner = build_relaxation_banner_html(retrieval)
    if banner:
        st.markdown(banner, unsafe_allow_html=True)

    if not candidates:
        st.error("No restaurants found even after relaxing constraints. Try a different location or cuisine.")
        return

    # ── LLM Recommendation Generation (skeleton loader while waiting) ───────────
    llm_message = f"{ai_label} is crafting personalised recommendations..."
    skeleton_slot = st.empty()
    skeleton_slot.markdown(
        build_skeleton_loader_html(count=3, message=llm_message),
        unsafe_allow_html=True,
    )
    try:
        result = connector.generate_recommendations(
            candidates=candidates,
            user_query=effective_query,
        )
    finally:
        skeleton_slot.empty()

    recommendations = result.get("recommendations", [])
    summary = result.get("summary", "")

    # Offline mode banner
    if "offline" in summary.lower() or "_API_KEY" in summary:
        st.markdown("""
        <div class="status-banner">
            🔌 <strong>Offline Mode:</strong> Results ranked by community ratings.
            Add your LLM API key to <code>.env</code> for AI-personalized recommendations.
        </div>
        """, unsafe_allow_html=True)

    # Results header
    st.markdown(f"""
    <div style="margin-bottom:1rem;">
        <span style="font-size:1.35rem; font-weight:700; color:#fff;">
            🎯 Top {len(recommendations)} Picks for "{location}"
        </span>
        <span style="font-size:0.8rem; color:rgba(255,255,255,0.35); margin-left:0.8rem;">
            filtered from {len(candidates)} candidates
        </span>
    </div>
    """, unsafe_allow_html=True)

    # AI Summary
    if summary:
        st.markdown(f'<div class="summary-block">💬 <strong>AI Summary:</strong> {summary}</div>', unsafe_allow_html=True)

    # Render recommendation cards
    for rec in recommendations:
        render_card(rec, rec.get("rank", 1))

    # Footer
    st.markdown(f"""
    <hr class="section-divider">
    <div style="text-align:center; color:rgba(255,255,255,0.2); font-size:0.75rem; padding-bottom:2rem;">
        Powered by {ai_label} • Data from Zomato • Built with Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
