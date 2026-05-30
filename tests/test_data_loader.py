"""Phase 1: data ingestion and cleaning tests."""

import pandas as pd
import pytest

from src.data_loader import ZomatoDataLoader


class TestZomatoDataLoader:
    def test_preprocess_required_columns_no_nulls(self, sample_raw_df, workspace_dir):
        loader = ZomatoDataLoader(workspace_dir=workspace_dir)
        df = loader.preprocess_data(sample_raw_df.copy())

        required = [
            "name",
            "cuisines",
            "locality",
            "city",
            "average_cost_for_two",
            "aggregate_rating",
            "votes",
        ]
        for col in required:
            assert col in df.columns
            assert df[col].notna().all(), f"Column {col} has null values"

    def test_rating_and_cost_are_numeric(self, sample_raw_df, workspace_dir):
        loader = ZomatoDataLoader(workspace_dir=workspace_dir)
        df = loader.preprocess_data(sample_raw_df.copy())

        assert pd.api.types.is_numeric_dtype(df["aggregate_rating"])
        assert pd.api.types.is_numeric_dtype(df["average_cost_for_two"])
        assert pd.api.types.is_integer_dtype(df["votes"])

    def test_cache_round_trip(self, cached_parquet, workspace_dir):
        loader = ZomatoDataLoader(workspace_dir=workspace_dir)
        df = loader.get_data(force_refresh=False)

        assert len(df) == 4
        assert cached_parquet.exists()

    def test_string_fields_trimmed(self, sample_raw_df, workspace_dir):
        raw = sample_raw_df.copy()
        raw.loc[0, "name"] = "  Spice Garden  "
        loader = ZomatoDataLoader(workspace_dir=workspace_dir)
        df = loader.preprocess_data(raw)
        spice_row = df[df["name"].str.contains("Spice Garden", na=False)]
        assert len(spice_row) == 1
        assert spice_row.iloc[0]["name"] == "Spice Garden"
