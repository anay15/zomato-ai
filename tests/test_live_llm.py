"""
Live LLM integration smoke test — calls the real Groq API using .env credentials.

Run: python -m pytest tests/test_live_llm.py -v -s
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from src.llm_connector import ZomatoLLMConnector

DUMMY_CANDIDATES = [
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


@pytest.mark.live
def test_live_groq_recommendation():
    """Calls Groq with dummy candidates; fails if offline fallback is used."""
    load_dotenv(ROOT / ".env", override=True)

    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if provider != "groq":
        pytest.skip(f"LLM_PROVIDER is '{provider}', not groq")
    if not api_key:
        pytest.skip(
            "GROQ_API_KEY is empty in .env on disk. "
            "Save your .env file (Ctrl+S) with your Groq key to run this test."
        )

    connector = ZomatoLLMConnector()
    assert connector._has_api_key(), "Groq API key not loaded from .env"

    result = connector.generate_recommendations(
        candidates=DUMMY_CANDIDATES,
        user_query="Lively birthday celebration with craft beer and finger food",
    )

    summary = result.get("summary", "")
    recommendations = result.get("recommendations", [])

    print("\n--- Live Groq Response ---")
    print(f"Provider: {connector.provider} | Model: {connector.groq_model}")
    print(f"Summary: {summary}")
    for rec in recommendations:
        print(f"  #{rec.get('rank')} {rec.get('name')}")
        print(f"     {rec.get('reasoning', '')[:200]}")

    assert recommendations, "Expected at least one recommendation from Groq"
    assert "offline" not in summary.lower(), (
        f"LLM call fell back to offline mode. Summary: {summary}"
    )
    assert all(rec.get("reasoning") for rec in recommendations), "Each rec needs reasoning"
