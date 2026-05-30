"""Shared pytest fixtures for phases 1–3."""

import pandas as pd
import pytest

from src.data_loader import ZomatoDataLoader
from src.llm_connector import ZomatoLLMConnector


@pytest.fixture
def workspace_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def sample_raw_df():
    """Minimal raw-style frame mimicking Hugging Face column names."""
    return pd.DataFrame(
        {
            "name": [
                "Spice Garden",
                "Connaught Cafe",
                "Budget Bites",
                "Premium Plate",
            ],
            "cuisines": [
                "North Indian, Mughlai",
                "Cafe, Italian",
                "North Indian",
                "French, Continental",
            ],
            "locality": [
                "Connaught Place",
                "Connaught Place, New Delhi",
                "Karol Bagh",
                "Saket",
            ],
            "city": ["New Delhi", "New Delhi", "New Delhi", "New Delhi"],
            "average_cost_for_two": ["800", "1,200", "350", "2,500"],
            "aggregate_rating": ["4.5/5", "4.2/5", "3.8", "4.9"],
            "votes": [1200, 800, 50, 300],
        }
    )


@pytest.fixture
def cleaned_df(sample_raw_df, workspace_dir):
    loader = ZomatoDataLoader(workspace_dir=workspace_dir)
    return loader.preprocess_data(sample_raw_df.copy())


@pytest.fixture
def cached_parquet(cleaned_df, workspace_dir):
    loader = ZomatoDataLoader(workspace_dir=workspace_dir)
    loader.data_dir.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_parquet(loader.cache_file, index=False)
    return loader.cache_file


@pytest.fixture
def dummy_candidates():
    return [
        {
            "name": "Byg Brewski Brewing Company",
            "locality": "Bellandur",
            "city": "Bangalore",
            "cuisines": "Finger Food, Burgers, American",
            "average_cost_for_two": 1200.0,
            "aggregate_rating": 4.9,
            "votes": 16345,
        },
        {
            "name": "AB's - Absolute Barbecues",
            "locality": "BTM",
            "city": "Bangalore",
            "cuisines": "BBQ, Continental, North Indian",
            "average_cost_for_two": 1100.0,
            "aggregate_rating": 4.9,
            "votes": 6375,
        },
        {
            "name": "Truffles",
            "locality": "BTM",
            "city": "Bangalore",
            "cuisines": "Burger, Italian, American",
            "average_cost_for_two": 900.0,
            "aggregate_rating": 4.7,
            "votes": 9100,
        },
    ]


@pytest.fixture
def llm_connector(monkeypatch):
    """Connector with no API keys so tests stay offline."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    return ZomatoLLMConnector()
