"""Phase 2: retrieval and filtering tests."""

import pandas as pd
import pytest

from src.retriever import RestaurantRetriever, RetrievalResult


@pytest.fixture
def retriever(cached_parquet, workspace_dir):
    return RestaurantRetriever(workspace_dir=workspace_dir)


class TestRestaurantRetriever:
    def test_location_substring_match(self, retriever):
        result = retriever.retrieve_candidates(
            location="Connaught",
            cuisines=[],
            budget_tier="",
            min_rating=None,
        )
        assert isinstance(result, RetrievalResult)
        assert result.status in (
            "strict",
            "relaxed_rating",
            "relaxed_budget",
            "nearby_locality",
            "fully_relaxed",
            "global_fallback",
        )
        assert len(result.candidates) > 0 or result.status == "global_fallback"

    def test_cuisine_token_match(self, retriever):
        result = retriever.retrieve_candidates(
            location="New Delhi",
            cuisines=["North Indian"],
            budget_tier="",
            min_rating=None,
        )
        assert not result.candidates.empty
        assert result.candidates["cuisines"].str.lower().str.contains("north indian").any()

    def test_budget_medium_filter(self, retriever):
        filtered = retriever._match_budget(retriever.df, "medium")
        assert not filtered.empty
        assert (filtered["average_cost_for_two"] >= 400).all()
        assert (filtered["average_cost_for_two"] <= 1000).all()

    def test_min_rating_filter(self, retriever):
        result = retriever.retrieve_candidates(
            location="New Delhi",
            cuisines=[],
            budget_tier="",
            min_rating=4.0,
        )
        if not result.candidates.empty and result.status == "strict":
            assert (result.candidates["aggregate_rating"] >= 4.0).all()

    def test_max_fifteen_candidates(self, retriever):
        big_df = pd.concat([retriever.df] * 5, ignore_index=True)
        retriever.df = big_df
        result = retriever.retrieve_candidates(location="New Delhi", cuisines=[])
        assert len(result.candidates) <= 15

    def test_trust_score_sorting(self, retriever):
        subset = retriever.df.head(5).copy()
        subset.loc[0, "aggregate_rating"] = 5.0
        subset.loc[0, "votes"] = 10
        subset.loc[1, "aggregate_rating"] = 4.0
        subset.loc[1, "votes"] = 50000

        sorted_df = retriever._sort_by_trust_score(subset)
        assert sorted_df.iloc[0]["votes"] >= sorted_df.iloc[-1]["votes"] or (
            sorted_df.iloc[0]["aggregate_rating"] >= sorted_df.iloc[-1]["aggregate_rating"]
        )

    def test_dataframe_to_records(self, retriever):
        result = retriever.retrieve_candidates(location="New Delhi", cuisines=[])
        records = RestaurantRetriever.dataframe_to_records(result.candidates)
        if records:
            assert "name" in records[0]
            assert "aggregate_rating" in records[0]

    def test_delhi_north_indian_medium_query(self, workspace_dir, cached_parquet):
        retriever = RestaurantRetriever(workspace_dir=workspace_dir)
        result = retriever.retrieve_candidates(
            location="Delhi",
            cuisines=["North Indian"],
            budget_tier="medium",
            min_rating=None,
        )
        assert len(result.candidates) <= 15
        assert result.status

    def test_retrieval_result_has_user_message_on_relaxation(self, retriever):
        result = retriever.retrieve_candidates(
            location="New Delhi",
            cuisines=["French"],
            budget_tier="low",
            min_rating=4.9,
        )
        if result.status != "strict":
            assert result.user_message
