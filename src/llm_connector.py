"""
Phase 3: Prompt Engineering & LLM Integration Layer
=====================================================
Bridges the retrieval engine with the LLM provider (Groq, Gemini, or OpenAI)
to generate personalized, structured restaurant recommendations.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from jsonschema import Draft7Validator, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

LLM_TEMPERATURE = 0.1

# Compact JSON contract sent to the LLM (minimizes token usage).
COMPACT_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["recs", "sum"],
    "properties": {
        "recs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "required": ["rk", "idx", "why"],
                "properties": {
                    "rk": {"type": "integer", "minimum": 1},
                    "idx": {"type": "integer", "minimum": 0},
                    "why": {"type": "string", "minLength": 1},
                },
            },
        },
        "sum": {"type": "string", "minLength": 1},
    },
}

# Normalized schema used after expanding compact LLM keys.
RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["recommendations", "summary"],
    "properties": {
        "recommendations": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "required": ["rank", "candidate_index", "name", "reasoning"],
                "properties": {
                    "rank": {"type": "integer", "minimum": 1},
                    "candidate_index": {"type": "integer", "minimum": 0},
                    "name": {"type": "string", "minLength": 1},
                    "reasoning": {"type": "string", "minLength": 1},
                },
            },
        },
        "summary": {"type": "string", "minLength": 1},
    },
}

_SCHEMA_VALIDATOR = Draft7Validator(RESPONSE_SCHEMA)
_COMPACT_VALIDATOR = Draft7Validator(COMPACT_RESPONSE_SCHEMA)

SYSTEM_PERSONA = (
    "You are an elite restaurant critic, local food expert, and personalized dining advisor "
    "inspired by the best of Zomato. You combine knowledge of cuisines, dining ambiance, "
    "budgets, and local neighborhoods to make precise, empathetic recommendations. "
    "You only recommend restaurants from the provided dataset. You NEVER invent details."
)

REPROMPT_SUFFIX = (
    "\n\nIMPORTANT: Your previous response was invalid or incomplete. "
    "Reply with ONLY a single valid JSON object matching the schema exactly. "
    "No markdown fences, no commentary."
)


class ZomatoLLMConnector:
    """
    Orchestrates LLM API calls to rank pre-filtered candidates and produce
    structured recommendations with evidence-based reasoning.
    """

    MAX_REPROMPT_ATTEMPTS = 1

    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
        self.groq_key = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()

        logger.info("LLM Connector initialized. Provider: '%s'", self.provider.upper())

        if self.provider == "groq" and not self.groq_key:
            logger.warning(
                "GROQ_API_KEY is not set in .env — live recommendations will use offline fallback."
            )
        elif self.provider == "gemini" and not self.gemini_key:
            logger.warning(
                "GEMINI_API_KEY is not set in .env — live recommendations will use offline fallback."
            )
        elif self.provider == "openai" and not self.openai_key:
            logger.warning(
                "OPENAI_API_KEY is not set in .env — live recommendations will use offline fallback."
            )

    def _has_api_key(self) -> bool:
        if self.provider == "groq":
            return bool(self.groq_key)
        if self.provider == "gemini":
            return bool(self.gemini_key)
        if self.provider == "openai":
            return bool(self.openai_key)
        return False

    def _api_key_env_var(self) -> str:
        if self.provider == "groq":
            return "GROQ_API_KEY"
        if self.provider == "gemini":
            return "GEMINI_API_KEY"
        if self.provider == "openai":
            return "OPENAI_API_KEY"
        return "LLM API key"

    def _build_prompt(self, candidates: List[Dict[str, Any]], user_query: str) -> str:
        slim_candidates = [
            {
                "i": idx,
                "n": c.get("name", "?"),
                "loc": c.get("locality", "?"),
                "cu": c.get("cuisines", ""),
                "c": c.get("average_cost_for_two", 0),
                "r": c.get("aggregate_rating", 0),
                "v": c.get("votes", 0),
            }
            for idx, c in enumerate(candidates)
        ]
        candidates_json = json.dumps(slim_candidates, separators=(",", ":"), ensure_ascii=False)

        return f"""{SYSTEM_PERSONA}

User preferences: "{user_query}"

Candidates (i=index, n=name, loc=locality, cu=cuisines, c=cost for two INR, r=rating, v=reviews):
{candidates_json}

Pick TOP 3-5 from the list only. Rank best first.
For each pick cite evidence from candidate fields in `why` (2-3 sentences).

Return STRICT JSON only:
{{"recs":[{{"rk":1,"idx":0,"why":"..."}}],"sum":"1-2 sentence overview"}}

Keys: rk=rank, idx=candidate index, why=reasoning, sum=summary.
"""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_groq(self, prompt: str) -> str:
        from groq import Groq

        if not self.groq_key:
            raise ValueError("GROQ_API_KEY is not configured in .env")

        client = Groq(api_key=self.groq_key)
        logger.info("Calling Groq model: %s...", self.groq_model)
        response = client.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_gemini(self, prompt: str) -> str:
        from google import genai
        from google.genai import types

        if not self.gemini_key:
            raise ValueError("GEMINI_API_KEY is not configured in .env")

        client = genai.Client(api_key=self.gemini_key)
        logger.info("Calling Gemini 2.0 Flash...")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=LLM_TEMPERATURE,
                response_mime_type="application/json",
                system_instruction=SYSTEM_PERSONA,
            ),
        )
        return response.text or ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_openai(self, prompt: str) -> str:
        from openai import OpenAI

        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY is not configured in .env")

        client = OpenAI(api_key=self.openai_key)
        logger.info("Calling OpenAI GPT-4o-mini...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    def _invoke_llm(self, prompt: str) -> str:
        if self.provider == "groq":
            return self._call_groq(prompt)
        if self.provider == "gemini":
            return self._call_gemini(prompt)
        if self.provider == "openai":
            return self._call_openai(prompt)
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{self.provider}'. Use 'groq', 'gemini', or 'openai'."
        )

    @staticmethod
    def _extract_json_object(raw_response: str) -> Dict[str, Any]:
        """Strip markdown fences and parse the top-level JSON object."""
        cleaned = raw_response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)

        return json.loads(cleaned)

    @staticmethod
    def _normalize_llm_payload(data: Dict[str, Any]) -> Dict[str, Any]:
        """Accept compact (`recs`/`sum`) or legacy (`recommendations`/`summary`) LLM JSON."""
        if "recs" in data:
            recommendations = []
            for rec in data.get("recs", []):
                idx = rec.get("idx", rec.get("candidate_index"))
                recommendations.append(
                    {
                        "rank": rec.get("rk", rec.get("rank", len(recommendations) + 1)),
                        "candidate_index": idx,
                        "name": rec.get("n", rec.get("name", "")),
                        "reasoning": rec.get("why", rec.get("reasoning", "")),
                    }
                )
            return {
                "recommendations": recommendations,
                "summary": data.get("sum", data.get("summary", "")),
            }
        return data

    def _validate_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "recs" in data:
            errors = sorted(_COMPACT_VALIDATOR.iter_errors(data), key=lambda e: e.path)
            if errors:
                messages = "; ".join(e.message for e in errors[:3])
                raise ValidationError(f"LLM JSON failed compact schema validation: {messages}")
            return self._normalize_llm_payload(data)

        errors = sorted(_SCHEMA_VALIDATOR.iter_errors(data), key=lambda e: e.path)
        if errors:
            messages = "; ".join(e.message for e in errors[:3])
            raise ValidationError(f"LLM JSON failed schema validation: {messages}")
        return data

    def _merge_with_candidates(
        self,
        data: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        validated: List[Dict[str, Any]] = []
        for rec in data.get("recommendations", []):
            idx = rec.get("candidate_index")
            if idx is None or not (0 <= idx < len(candidates)):
                logger.warning(
                    "Discarding invalid candidate_index=%s (valid: 0–%s)",
                    idx,
                    len(candidates) - 1,
                )
                continue

            full = candidates[idx]
            validated.append(
                {
                    "rank": rec.get("rank", len(validated) + 1),
                    "name": full.get("name"),
                    "locality": full.get("locality"),
                    "city": full.get("city"),
                    "cuisines": full.get("cuisines"),
                    "average_cost_for_two": full.get("average_cost_for_two"),
                    "aggregate_rating": full.get("aggregate_rating"),
                    "votes": full.get("votes"),
                    "reasoning": rec.get(
                        "reasoning", "A great match for your preferences."
                    ),
                }
            )

        return {
            "recommendations": validated,
            "summary": data.get("summary", "Top restaurant picks based on your search."),
        }

    def _parse_and_validate(
        self,
        raw_response: str,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        data = self._extract_json_object(raw_response)
        data = self._validate_schema(data)
        return self._merge_with_candidates(data, candidates)

    def _offline_fallback(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("Using offline fallback: ranking by rating and vote count.")
        fallback = []
        for idx, c in enumerate(candidates[:3]):
            fallback.append(
                {
                    "rank": idx + 1,
                    "name": c.get("name"),
                    "locality": c.get("locality"),
                    "city": c.get("city"),
                    "cuisines": c.get("cuisines"),
                    "average_cost_for_two": c.get("average_cost_for_two"),
                    "aggregate_rating": c.get("aggregate_rating"),
                    "votes": c.get("votes"),
                    "reasoning": (
                        f"Ranked by community trust: rated {c.get('aggregate_rating')}/5 "
                        f"by {c.get('votes')} reviews. A reliable choice in {c.get('locality')}."
                    ),
                }
            )
        return {
            "recommendations": fallback,
            "summary": (
                f"Showing top-rated matches (offline mode — add your {self._api_key_env_var()} "
                "to .env for AI-personalized recommendations)."
            ),
        }

    def _call_with_reprompt(
        self,
        prompt: str,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Call LLM once; on parse/schema failure, re-prompt once with stricter instructions."""
        last_error: Optional[Exception] = None
        current_prompt = prompt

        for attempt in range(self.MAX_REPROMPT_ATTEMPTS + 1):
            raw = self._invoke_llm(current_prompt)
            try:
                return self._parse_and_validate(raw, candidates)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "LLM response invalid (attempt %s/%s): %s",
                    attempt + 1,
                    self.MAX_REPROMPT_ATTEMPTS + 1,
                    exc,
                )
                if attempt < self.MAX_REPROMPT_ATTEMPTS:
                    current_prompt = prompt + REPROMPT_SUFFIX
                else:
                    logger.error("Re-prompt exhausted. Raw snippet: %s", raw[:500])

        raise last_error or RuntimeError("LLM response parsing failed")

    def generate_recommendations(
        self,
        candidates: List[Dict[str, Any]],
        user_query: str,
    ) -> Dict[str, Any]:
        """
        Full Phase 3 pipeline: prompt → LLM → schema validation → candidate merge,
        with re-prompt and offline fallbacks.
        """
        if not candidates:
            return {
                "recommendations": [],
                "summary": (
                    "No restaurants matched your filters. "
                    "Try broadening your search criteria."
                ),
            }

        if not self._has_api_key():
            logger.warning("No API key found. Returning offline fallback recommendations.")
            return self._offline_fallback(candidates)

        prompt = self._build_prompt(candidates, user_query)

        try:
            result = self._call_with_reprompt(prompt, candidates)
            if not result["recommendations"]:
                logger.warning("Parsed response had 0 valid recommendations. Using fallback.")
                return self._offline_fallback(candidates)
            return result
        except Exception as exc:
            logger.error("LLM API call failed (%s): %s", self.provider.upper(), exc)
            return self._offline_fallback(candidates)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 3 LLM Connector — Verification Test")
    logger.info("=" * 60)

    connector = ZomatoLLMConnector()

    dummy_candidates = [
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

    result = connector.generate_recommendations(
        candidates=dummy_candidates,
        user_query=(
            "A lively place to celebrate a birthday with friends, "
            "preferably with craft beer and finger food"
        ),
    )

    print("\n--- Recommendations Output ---")
    print(json.dumps(result, indent=2, ensure_ascii=True))

    empty_result = connector.generate_recommendations(candidates=[], user_query="Anything")
    print(f"\n--- Empty Candidates: {empty_result['summary']} ---")
