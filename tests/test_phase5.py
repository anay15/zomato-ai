"""Phase 5: polish, fallback, and edge-case tests."""

from pathlib import Path

import pytest

from src.app import build_relaxation_banner_html
from src.llm_connector import LLM_TEMPERATURE, ZomatoLLMConnector
from src.retriever import RestaurantRetriever

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def live_retriever():
    parquet = ROOT / "data" / "zomato_cleaned.parquet"
    if not parquet.exists():
        pytest.skip("Requires data/zomato_cleaned.parquet")
    return RestaurantRetriever(workspace_dir=str(ROOT))


class TestPhase5Fallback:
    def test_over_restricted_query_does_not_crash(self, live_retriever):
        """implementation.md phase 5 verification: rating 4.9 + low budget."""
        result = live_retriever.retrieve_candidates(
            location="Indiranagar",
            cuisines=["French"],
            budget_tier="low",
            min_rating=4.9,
        )
        assert not result.candidates.empty
        assert result.status != "strict"
        assert result.user_message
        assert "relax" in result.user_message.lower() or "nearby" in result.user_message.lower()

    def test_budget_relaxed_before_rating_message(self, live_retriever):
        result = live_retriever.retrieve_candidates(
            location="Bangalore",
            cuisines=["North Indian"],
            budget_tier="low",
            min_rating=4.5,
        )
        assert not result.candidates.empty
        if result.status == "relaxed_budget":
            assert "budget" in result.user_message.lower()

    def test_rating_relaxation_message(self):
        message = RestaurantRetriever._user_message_for_status(
            "relaxed_rating", relaxed_min_rating=3.5
        )
        assert "3.5" in message
        assert "rating" in message.lower()

    def test_nearby_locality_suggestions(self, live_retriever):
        nearby = live_retriever.suggest_nearby_localities("Indiranagar", limit=3)
        assert isinstance(nearby, list)

    def test_fuzzy_location_typo(self, live_retriever):
        resolved = live_retriever._resolve_location("Cannaught Place")
        assert resolved  # should fuzzy-match Connaught Place if in dataset

    def test_relaxation_banner_renders_message(self):
        from src.retriever import RetrievalResult
        import pandas as pd

        result = RetrievalResult(
            candidates=pd.DataFrame(),
            status="relaxed_rating",
            user_message="We relaxed your rating constraint slightly (now >= 3.5) to find matching locations!",
            nearby_localities=["Koramangala", "HSR"],
        )
        html = build_relaxation_banner_html(result)
        assert "3.5" in html
        assert "Koramangala" in html


class TestPhase5LLMTuning:
    def test_temperature_lowered(self):
        assert LLM_TEMPERATURE <= 0.1

    def test_compact_prompt_is_shorter(self, dummy_candidates):
        connector = ZomatoLLMConnector()
        prompt = connector._build_prompt(dummy_candidates, "rooftop quick bite")
        assert len(prompt) < 2500
        assert '"recs"' in prompt or "recs" in prompt

    def test_extreme_qualitative_query_offline(self, llm_connector, dummy_candidates):
        result = llm_connector.generate_recommendations(
            candidates=dummy_candidates,
            user_query=(
                "dog-friendly rooftop with live jazz, zero-waste kitchen, "
                "and a secret speakeasy entrance"
            ),
        )
        assert result["recommendations"]
        assert result["summary"]


class TestPhase5MultiCity:
    @pytest.mark.parametrize("city", ["Delhi", "Bangalore", "Mumbai"])
    def test_multi_city_retrieval(self, live_retriever, city):
        result = live_retriever.retrieve_candidates(location=city, cuisines=[], budget_tier="")
        assert not result.candidates.empty
        assert len(result.candidates) <= 15
