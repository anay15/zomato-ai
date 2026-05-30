"""Phase 3: prompt engineering and LLM integration tests."""

import json

import pytest
from jsonschema import ValidationError

from src.llm_connector import RESPONSE_SCHEMA, ZomatoLLMConnector


VALID_LLM_JSON = json.dumps(
    {
        "recs": [
            {
                "rk": 1,
                "idx": 0,
                "why": "Great craft beer and finger food for group celebrations in Bellandur.",
            }
        ],
        "sum": "Top pick for a lively birthday with craft beer.",
    }
)

LEGACY_LLM_JSON = json.dumps(
    {
        "recommendations": [
            {
                "rank": 1,
                "candidate_index": 0,
                "name": "Byg Brewski Brewing Company",
                "reasoning": "Great craft beer and finger food for group celebrations in Bellandur.",
            }
        ],
        "summary": "Top pick for a lively birthday with craft beer.",
    }
)


class TestZomatoLLMConnector:
    def test_offline_fallback_parses_to_dict(self, llm_connector, dummy_candidates):
        result = llm_connector.generate_recommendations(
            candidates=dummy_candidates,
            user_query="craft beer birthday party",
        )
        assert isinstance(result, dict)
        assert "recommendations" in result
        assert "summary" in result
        assert len(result["recommendations"]) >= 1
        first = result["recommendations"][0]
        assert first["name"] == dummy_candidates[0]["name"]
        assert "reasoning" in first

    def test_empty_candidates_guard(self, llm_connector):
        result = llm_connector.generate_recommendations(candidates=[], user_query="anything")
        assert result["recommendations"] == []
        assert "No restaurants matched" in result["summary"]

    def test_build_prompt_includes_query_and_candidates(self, llm_connector, dummy_candidates):
        prompt = llm_connector._build_prompt(dummy_candidates, "rooftop dining")
        assert "rooftop dining" in prompt
        assert "Byg Brewski Brewing Company" in prompt
        assert '"i":0' in prompt or '"i": 0' in prompt

    def test_compact_json_normalization(self, llm_connector):
        normalized = llm_connector._normalize_llm_payload(json.loads(VALID_LLM_JSON))
        assert normalized["recommendations"][0]["candidate_index"] == 0
        assert normalized["summary"]

    def test_extract_json_strips_markdown_fences(self):
        wrapped = f"```json\n{VALID_LLM_JSON}\n```"
        data = ZomatoLLMConnector._extract_json_object(wrapped)
        assert data["sum"]

    def test_validate_schema_rejects_invalid_payload(self, llm_connector):
        with pytest.raises(ValidationError):
            llm_connector._validate_schema({"recs": [], "sum": "x"})

    def test_parse_and_validate_merges_candidate_fields(
        self, llm_connector, dummy_candidates
    ):
        result = llm_connector._parse_and_validate(VALID_LLM_JSON, dummy_candidates)
        assert result["recommendations"][0]["locality"] == "Bellandur"
        assert result["recommendations"][0]["aggregate_rating"] == 4.9

    def test_parse_and_validate_accepts_legacy_json(self, llm_connector, dummy_candidates):
        result = llm_connector._parse_and_validate(LEGACY_LLM_JSON, dummy_candidates)
        assert result["recommendations"][0]["locality"] == "Bellandur"

    def test_hallucinated_index_discarded(self, llm_connector, dummy_candidates):
        bad = json.dumps(
            {
                "recs": [
                    {"rk": 1, "idx": 99, "why": "Should be dropped."},
                    {"rk": 2, "idx": 1, "why": "Valid pick."},
                ],
                "sum": "Mixed validity.",
            }
        )
        result = llm_connector._parse_and_validate(bad, dummy_candidates)
        assert len(result["recommendations"]) == 1
        assert result["recommendations"][0]["name"] == "AB's - Absolute Barbecues"

    def test_reprompt_on_malformed_json(self, llm_connector, dummy_candidates, monkeypatch):
        calls = []

        def fake_invoke(prompt):
            calls.append(prompt)
            if len(calls) == 1:
                return "not valid json at all"
            return VALID_LLM_JSON

        monkeypatch.setattr(llm_connector, "_has_api_key", lambda: True)
        monkeypatch.setattr(llm_connector, "_invoke_llm", fake_invoke)

        result = llm_connector.generate_recommendations(
            candidates=dummy_candidates,
            user_query="quiet dinner",
        )
        assert len(calls) == 2
        assert "IMPORTANT" in calls[1]
        assert len(result["recommendations"]) >= 1

    def test_live_path_uses_mocked_llm(self, llm_connector, dummy_candidates, monkeypatch):
        monkeypatch.setattr(llm_connector, "_has_api_key", lambda: True)
        monkeypatch.setattr(llm_connector, "_invoke_llm", lambda prompt: VALID_LLM_JSON)

        result = llm_connector.generate_recommendations(
            candidates=dummy_candidates,
            user_query="dog-friendly patio",
        )
        assert result["recommendations"][0]["reasoning"]
        assert "dog-friendly" not in result["recommendations"][0]["name"]

    def test_groq_provider_routing(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        connector = ZomatoLLMConnector()
        assert connector.provider == "groq"
        assert connector._has_api_key() is True

        called = {"groq": False}

        def fake_groq(prompt):
            called["groq"] = True
            return VALID_LLM_JSON

        monkeypatch.setattr(connector, "_call_groq", fake_groq)
        connector._invoke_llm("test prompt")
        assert called["groq"] is True

    def test_response_schema_contract(self):
        assert RESPONSE_SCHEMA["required"] == ["recommendations", "summary"]
