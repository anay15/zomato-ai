import logging
import sys
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_loader import ZomatoDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MIN_CANDIDATES = 3


@dataclass
class RetrievalResult:
    """Structured retrieval output for UI and downstream LLM handoff."""

    candidates: pd.DataFrame
    status: str
    user_message: str
    nearby_localities: List[str] = field(default_factory=list)
    relaxed_min_rating: Optional[float] = None


class RestaurantRetriever:
    """
    Core retrieval and pre-filtering engine that queries the cleaned Zomato dataset.
    Phase 5: progressive relaxation — budget → rating → nearby localities → global.
    """

    BUDGET_RANGES = {
        "low": (0, 400),
        "medium": (400, 1000),
        "high": (1000, 999999),
    }

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.data_loader = ZomatoDataLoader(workspace_dir=self.workspace_dir)
        self.df: Optional[pd.DataFrame] = None
        self._known_locations: List[str] = []
        self.load_data()

    def load_data(self) -> None:
        try:
            self.df = self.data_loader.get_data()
            self._known_locations = self._build_location_index()
            logger.info("Retriever initialized with %s restaurants.", len(self.df))
        except Exception as exc:
            logger.error("Failed to load dataset in retriever: %s", exc)
            raise

    def _build_location_index(self) -> List[str]:
        if self.df is None or self.df.empty:
            return []
        localities = self.df["locality"].dropna().astype(str).unique().tolist()
        cities = self.df["city"].dropna().astype(str).unique().tolist()
        return sorted(set(localities + cities))

    def _resolve_location(self, location_query: str) -> str:
        """Map typos to the closest known locality/city name."""
        if not location_query or not self._known_locations:
            return location_query
        query = location_query.strip()
        if not query:
            return location_query
        if any(query.lower() == loc.lower() for loc in self._known_locations):
            return query
        match = get_close_matches(query, self._known_locations, n=1, cutoff=0.75)
        if match:
            logger.info("Fuzzy-matched location '%s' → '%s'", query, match[0])
            return match[0]
        return query

    def _match_location(self, df: pd.DataFrame, location_query: str) -> pd.DataFrame:
        if not location_query or not isinstance(location_query, str):
            return df
        query = location_query.strip().lower()
        loc_mask = df["locality"].str.lower().str.contains(query, na=False)
        city_mask = df["city"].str.lower().str.contains(query, na=False)
        return df[loc_mask | city_mask]

    def _match_locations(self, df: pd.DataFrame, locations: List[str]) -> pd.DataFrame:
        if not locations:
            return df
        lowered = {loc.strip().lower() for loc in locations if loc.strip()}
        loc_mask = df["locality"].str.lower().isin(lowered)
        city_mask = df["city"].str.lower().isin(lowered)
        contains_mask = df["locality"].str.lower().apply(
            lambda value: any(q in value for q in lowered)
        )
        return df[loc_mask | city_mask | contains_mask]

    def _match_cuisines(self, df: pd.DataFrame, cuisines_query: List[str]) -> pd.DataFrame:
        if not cuisines_query:
            return df
        queries = [c.strip().lower() for c in cuisines_query if c.strip()]
        if not queries:
            return df

        def has_cuisine_intersection(cuisine_str: str) -> bool:
            if pd.isna(cuisine_str):
                return False
            restaurant_cuisines = [c.strip().lower() for c in cuisine_str.split(",")]
            for q in queries:
                if any(q in rc or rc in q for rc in restaurant_cuisines):
                    return True
            return False

        mask = df["cuisines"].apply(has_cuisine_intersection).astype(bool)
        return df[mask]

    def _match_budget(self, df: pd.DataFrame, budget_tier: str) -> pd.DataFrame:
        if not budget_tier or not isinstance(budget_tier, str):
            return df
        tier = budget_tier.strip().lower()
        if tier not in self.BUDGET_RANGES:
            logger.warning("Unknown budget tier: '%s'. Ignoring budget filter.", budget_tier)
            return df
        min_cost, max_cost = self.BUDGET_RANGES[tier]
        return df[
            (df["average_cost_for_two"] >= min_cost)
            & (df["average_cost_for_two"] <= max_cost)
        ]

    def _match_rating(self, df: pd.DataFrame, min_rating: float) -> pd.DataFrame:
        if min_rating is None:
            return df
        return df[df["aggregate_rating"] >= min_rating]

    def _sort_by_trust_score(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        scored = df.copy()
        scored["_trust_score"] = scored["aggregate_rating"] * np.log1p(scored["votes"])
        return (
            scored.sort_values("_trust_score", ascending=False)
            .drop(columns="_trust_score")
            .reset_index(drop=True)
        )

    def _finalize_candidates(self, df: pd.DataFrame, max_candidates: int) -> pd.DataFrame:
        return self._sort_by_trust_score(df).head(max_candidates)

    def _city_for_location(self, location: str) -> Optional[str]:
        if self.df is None or not location:
            return None
        matches = self._match_location(self.df, location)
        if matches.empty:
            fuzzy = self._resolve_location(location)
            matches = self._match_location(self.df, fuzzy)
        if matches.empty:
            return None
        return matches["city"].mode().iloc[0]

    def suggest_nearby_localities(self, location: str, limit: int = 5) -> List[str]:
        """Return popular localities in the same city as the queried location."""
        if self.df is None or self.df.empty or not location:
            return []

        city = self._city_for_location(location)
        pool = self.df[self.df["city"] == city] if city else self._match_location(self.df, location)
        if pool.empty:
            return []

        query_lower = location.strip().lower()
        locality_counts = (
            pool.groupby("locality")
            .size()
            .sort_values(ascending=False)
        )
        nearby = [
            loc
            for loc in locality_counts.index.astype(str).tolist()
            if loc.lower() != query_lower and query_lower not in loc.lower()
        ]
        return nearby[:limit]

    def _apply_base_filters(
        self,
        df: pd.DataFrame,
        location: str,
        cuisines: Optional[List[str]],
        budget_tier: str,
        min_rating: Optional[float],
        extra_locations: Optional[List[str]] = None,
        apply_budget: bool = True,
    ) -> pd.DataFrame:
        result = df.copy()
        locs = [location] + (extra_locations or [])
        result = self._match_locations(result, locs)
        result = self._match_cuisines(result, cuisines)
        if apply_budget:
            result = self._match_budget(result, budget_tier)
        result = self._match_rating(result, min_rating)
        return result

    @staticmethod
    def _user_message_for_status(
        status: str,
        relaxed_min_rating: Optional[float] = None,
        nearby_localities: Optional[List[str]] = None,
    ) -> str:
        if status == "strict":
            return ""
        if status == "relaxed_budget":
            return (
                "We relaxed your budget constraint slightly to find matching restaurants!"
            )
        if status == "relaxed_rating":
            rating_text = f"{relaxed_min_rating:.1f}" if relaxed_min_rating else "3.5"
            return (
                f"We relaxed your rating constraint slightly (now >= {rating_text}) "
                "to find matching locations!"
            )
        if status == "nearby_locality":
            nearby = ", ".join(nearby_localities[:3]) if nearby_localities else "nearby areas"
            return (
                f"No exact matches in your area. We expanded the search to nearby "
                f"localities: {nearby}."
            )
        if status == "global_fallback":
            return (
                "No restaurants matched your filters. Showing top-rated options from "
                "our full database."
            )
        if status == "empty_dataset":
            return "The restaurant database is unavailable right now."
        return "We broadened your search to find the best available options."

    @staticmethod
    def dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
        if df is None or df.empty:
            return []
        cols = [
            "name",
            "locality",
            "city",
            "cuisines",
            "average_cost_for_two",
            "aggregate_rating",
            "votes",
        ]
        present = [c for c in cols if c in df.columns]
        return df[present].to_dict(orient="records")

    def retrieve_candidates(
        self,
        location: str = "",
        cuisines: List[str] = None,
        budget_tier: str = "",
        min_rating: float = None,
        max_candidates: int = 15,
    ) -> RetrievalResult:
        """
        Progressive relaxation when strict filters yield too few matches:
        1. Strict filters
        2. Relax budget
        3. Relax rating (e.g. 4.0 → 3.5)
        4. Expand to nearby localities in the same city
        5. Global top-rated fallback
        """
        if self.df is None or self.df.empty:
            logger.warning("Empty dataset. Cannot retrieve candidates.")
            return RetrievalResult(
                candidates=pd.DataFrame(),
                status="empty_dataset",
                user_message=RestaurantRetriever._user_message_for_status("empty_dataset"),
            )

        resolved_location = self._resolve_location(location)
        cuisines = cuisines or []
        has_budget = bool(budget_tier and budget_tier.strip())
        has_rating = min_rating is not None

        logger.info(
            "Applying strict filters: Location='%s', Cuisines=%s, Budget='%s', Rating=%s",
            resolved_location,
            cuisines,
            budget_tier,
            min_rating,
        )

        strict = self._apply_base_filters(
            self.df,
            resolved_location,
            cuisines,
            budget_tier,
            min_rating,
            apply_budget=True,
        )
        if len(strict) >= MIN_CANDIDATES:
            logger.info("Strict search returned %s matches.", len(strict))
            return RetrievalResult(
                candidates=self._finalize_candidates(strict, max_candidates),
                status="strict",
                user_message="",
            )

        relaxed_min_rating = (
            max(3.0, min_rating - 0.5) if min_rating is not None else 3.0
        )

        # Step 1: Relax budget first (keep location, cuisine, original rating)
        if has_budget:
            logger.info("Relaxing budget constraints...")
            relax_budget = self._apply_base_filters(
                self.df,
                resolved_location,
                cuisines,
                budget_tier,
                min_rating,
                apply_budget=False,
            )
            if len(relax_budget) >= MIN_CANDIDATES:
                logger.info("Budget relaxation resolved search with %s matches.", len(relax_budget))
                return RetrievalResult(
                    candidates=self._finalize_candidates(relax_budget, max_candidates),
                    status="relaxed_budget",
                    user_message=RestaurantRetriever._user_message_for_status("relaxed_budget"),
                    relaxed_min_rating=min_rating,
                )

        # Step 2: Relax rating cutoff
        if has_rating and relaxed_min_rating < min_rating:
            logger.info("Relaxing rating constraint to >= %s...", relaxed_min_rating)
            relax_rating = self._apply_base_filters(
                self.df,
                resolved_location,
                cuisines,
                budget_tier,
                relaxed_min_rating,
                apply_budget=False,
            )
            if len(relax_rating) >= MIN_CANDIDATES:
                logger.info("Rating relaxation resolved search with %s matches.", len(relax_rating))
                return RetrievalResult(
                    candidates=self._finalize_candidates(relax_rating, max_candidates),
                    status="relaxed_rating",
                    user_message=RestaurantRetriever._user_message_for_status(
                        "relaxed_rating", relaxed_min_rating=relaxed_min_rating
                    ),
                    relaxed_min_rating=relaxed_min_rating,
                )

        # Step 3: Expand to nearby localities in the same city
        nearby = self.suggest_nearby_localities(resolved_location)
        if nearby:
            logger.info("Expanding search to nearby localities: %s", nearby[:3])
            nearby_matches = self._apply_base_filters(
                self.df,
                resolved_location,
                cuisines,
                budget_tier,
                relaxed_min_rating,
                extra_locations=nearby,
                apply_budget=False,
            )
            if len(nearby_matches) >= MIN_CANDIDATES or not nearby_matches.empty:
                logger.info("Nearby locality search returned %s matches.", len(nearby_matches))
                return RetrievalResult(
                    candidates=self._finalize_candidates(nearby_matches, max_candidates),
                    status="nearby_locality",
                    user_message=RestaurantRetriever._user_message_for_status(
                        "nearby_locality", nearby_localities=nearby
                    ),
                    nearby_localities=nearby,
                    relaxed_min_rating=relaxed_min_rating,
                )

        # Step 4: Location-only with floor rating
        location_only = self._match_location(self.df, resolved_location)
        location_only = self._match_rating(location_only, 3.0)
        if not location_only.empty:
            logger.info("Fully relaxed location search returned %s matches.", len(location_only))
            return RetrievalResult(
                candidates=self._finalize_candidates(location_only, max_candidates),
                status="fully_relaxed",
                user_message=(
                    "We relaxed cuisine, budget, and rating filters to show options "
                    f"in and around {resolved_location}."
                ),
                nearby_localities=nearby,
                relaxed_min_rating=3.0,
            )

        # Step 5: Global fallback
        logger.info("Zero location matches. Returning top-rated restaurants overall.")
        return RetrievalResult(
            candidates=self._finalize_candidates(self.df, max_candidates),
            status="global_fallback",
            user_message=RestaurantRetriever._user_message_for_status("global_fallback"),
            nearby_localities=nearby,
        )


if __name__ == "__main__":
    logger.info("Starting Restaurant Retriever test run...")
    retriever = RestaurantRetriever()

    strict_result = retriever.retrieve_candidates(
        location="BTM",
        cuisines=["Italian", "Cafe"],
        budget_tier="Medium",
        min_rating=4.0,
    )
    print(f"\n--- Strict Query Status: {strict_result.status} ---")
    print(strict_result.user_message)
    print(
        strict_result.candidates[
            ["name", "city", "locality", "cuisines", "average_cost_for_two", "aggregate_rating"]
        ].head(5)
    )

    relaxed_result = retriever.retrieve_candidates(
        location="Indiranagar",
        cuisines=["French"],
        budget_tier="Low",
        min_rating=4.9,
    )
    print(f"\n--- Relaxed Query Status: {relaxed_result.status} ---")
    print(relaxed_result.user_message)
    print(f"Nearby suggestions: {relaxed_result.nearby_localities[:5]}")
    print(
        relaxed_result.candidates[
            ["name", "city", "locality", "cuisines", "average_cost_for_two", "aggregate_rating"]
        ].head(5)
    )
