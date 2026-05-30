"""Run a single live Groq API smoke test (uses .env credentials)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def main() -> int:
    connector = ZomatoLLMConnector()

    print("Provider:", connector.provider)
    print("Model:", connector.groq_model)
    print("API key loaded:", connector._has_api_key())

    if not connector._has_api_key():
        print("ERROR: GROQ_API_KEY not found. Save your .env file and try again.")
        return 1

    print("\nCalling Groq LLM...\n")
    result = connector.generate_recommendations(
        candidates=DUMMY_CANDIDATES,
        user_query="Lively birthday celebration with craft beer and finger food",
    )

    summary = result.get("summary", "")
    recommendations = result.get("recommendations", [])

    if "offline" in summary.lower() or not recommendations:
        print("FAILED — fell back to offline mode or empty response")
        print("Summary:", summary)
        return 1

    print("SUCCESS — live Groq call worked!\n")
    print("Summary:", summary)
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"  #{rec.get('rank')} {rec.get('name')}")
        print(f"     {rec.get('reasoning', '')}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
