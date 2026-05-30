"""Phase 4: Streamlit UI helper tests."""

import pandas as pd

from src.app import (
    build_card_html,
    build_skeleton_loader_html,
    budget_label,
    filter_locations,
    get_unique_cuisines,
    get_unique_locations,
    provider_display_name,
    rank_class,
    rating_class,
)


class TestAppHelpers:
    def test_rating_class_tiers(self):
        assert rating_class(4.8) == "rating-high"
        assert rating_class(4.0) == "rating-mid"
        assert rating_class(3.0) == "rating-low"

    def test_rank_class(self):
        assert rank_class(1) == "rank-1"
        assert rank_class(4) == "rank-other"

    def test_budget_label(self):
        assert budget_label(300) == "Budget-Friendly"
        assert budget_label(700) == "Moderate"
        assert budget_label(1500) == "Premium"

    def test_provider_display_name(self):
        assert provider_display_name("groq") == "Groq AI"
        assert provider_display_name("unknown") == "AI"

    def test_filter_locations(self):
        locs = ["Connaught Place", "Karol Bagh", "Indiranagar"]
        assert filter_locations(locs, "conn") == ["Connaught Place"]
        assert filter_locations(locs, "") == locs

    def test_get_unique_locations(self, cleaned_df):
        locs = get_unique_locations(cleaned_df)
        assert "Connaught Place" in locs
        assert "Unknown Locality" not in locs

    def test_get_unique_cuisines(self, cleaned_df):
        cuisines = get_unique_cuisines(cleaned_df)
        assert "North Indian" in cuisines
        assert "Multi-Cuisine" not in cuisines

    def test_build_card_html_contains_key_fields(self):
        rec = {
            "name": "Spice Garden",
            "locality": "Connaught Place",
            "city": "New Delhi",
            "cuisines": "North Indian, Mughlai",
            "average_cost_for_two": 800,
            "aggregate_rating": 4.5,
            "votes": 1200,
            "reasoning": "Perfect for a hearty North Indian dinner.",
        }
        html = build_card_html(rec, rank=1)
        assert "Spice Garden" in html
        assert "AI Insight" in html
        assert "rating-high" in html
        assert "₹800" in html

    def test_skeleton_loader_html(self):
        html = build_skeleton_loader_html(count=2, message="Loading...")
        assert "skeleton-card" in html
        assert html.count("skeleton-card") == 2
        assert "Loading..." in html
        assert "skeleton-spinner" in html

    def test_app_module_imports(self):
        import src.app as app_module

        assert callable(app_module.main)
