# Edge-Case Handling Strategy

This document catalogs critical edge cases that may occur across all phases of the **AI-Powered Restaurant Recommendation System** and provides robust technical mitigation strategies for each.

---

## 🗺️ Edge-Case Matrix Overview

| System Layer | Edge Case Scenario | Impact Level | Primary Mitigation Strategy |
| :--- | :--- | :---: | :--- |
| **Data Ingestion** | Hugging Face Dataset Offline / Network Timeout | 🔴 High | Local Parquet caching, automated retries, and offline mock-data fallback. |
| **Data Cleaning** | Malformed, empty, or anomalous numerical fields | 🟡 Medium | Hard constraint schema validation and robust regex parsing. |
| **Retrieval Engine** | Zero-match search filter results | 🔴 High | Progressive constraint relaxation flow (budget, rating, locality). |
| **Retrieval Engine** | Location/Cuisine spelling variations and typos | 🟡 Medium | Fuzzy matching and token-based substring intersection mapping. |
| **LLM Connector** | Rate limits (HTTP 429) or token limits exceeded | 🟡 Medium | Strict candidate count caps, exponential backoff, and model fallback. |
| **LLM Connector** | Hallucinated outputs (recommending non-existent data) | 🔴 High | Candidate ID verification against local databases before rendering. |
| **LLM Connector** | Malformed JSON or structural generation parser crash | 🔴 High | Strict JSON Mode enforcement, Pydantic validation, and clean formatting retries. |
| **UI Presentation** | Blank/empty user inputs or extremely specific niche queries | 🟢 Low | Smart default states, placeholder text prompts, and robust UI alerts. |

---

## 🛠️ Detailed Edge-Cases & Mitigation Blueprints

### 1. Data Ingestion & Preprocessing Phase

#### 🔴 Scenario A: Hugging Face Hub Offline or Network Failures
* **Problem**: The app fails to boot or crashes when loading the dataset `ManikaSaini/zomato-restaurant-recommendation` at startup because of network issues or Hugging Face server downtime.
* **Mitigation Strategy**:
  * **Local Parquet Caching**: `data_loader.py` must first check if `data/zomato_cleaned.parquet` exists locally. If present, load it instantly without accessing the network.
  * **Fallback Mock Dataset**: Store a tiny, embedded backup file (`data/mock_restaurants.parquet` containing 50 popular restaurants) to allow the app to operate in "offline demonstration mode" if both local caching and network connection fail.

#### 🟡 Scenario B: Malformed Fields (e.g., Cost = 0, Cuisines = None, NaN Ratings)
* **Problem**: Certain dataset records contain null, zero, or text values in fields meant for numeric filtering (like average cost or rating).
* **Mitigation Strategy**:
  * Cleanse inputs during data preparation:
    ```python
    # Force average cost to be numeric, replace zeros/nans with median cost of locality
    df['Average_Cost_for_Two'] = pd.to_numeric(df['Average_Cost_for_Two'], errors='coerce')
    median_cost = df['Average_Cost_for_Two'].median()
    df['Average_Cost_for_Two'] = df['Average_Cost_for_Two'].fillna(median_cost).replace(0, median_cost)

    # Standardize rating, mapping non-numeric (e.g., 'NEW', '-') to a default low rating (2.5)
    df['Aggregate_Rating'] = pd.to_numeric(df['Aggregate_Rating'], errors='coerce').fillna(2.5)
    ```

---

### 2. Retrieval & Pre-Filtering Engine Phase

#### 🔴 Scenario C: The "Zero Match" Query (Over-Restricted Filters)
* **Problem**: The user enters highly restrictive criteria (e.g., Location: "Noida", Cuisine: "French", Budget: "Low", Min Rating: 4.8), resulting in an empty pre-filtered DataFrame. No candidates are sent to the LLM.
* **Mitigation Strategy**:
  * Implement a **Progressive Relaxation Loop** in `retriever.py` that broadens search bounds sequentially if matches fall below `3` restaurants:
    ```python
    def get_candidates_with_fallback(filters, df):
        candidates = apply_strict_filters(filters, df)
        if len(candidates) >= 3:
            return candidates, "strict"
        
        # Step 1: Relax Rating threshold by 0.5 points
        candidates = apply_filters(filters, df, relaxed_rating=True)
        if len(candidates) >= 3:
            return candidates, "relaxed_rating"
        
        # Step 2: Relax Budget constraints (expand range bounds by 50%)
        candidates = apply_filters(filters, df, relaxed_rating=True, relaxed_budget=True)
        if len(candidates) >= 3:
            return candidates, "relaxed_budget"
        
        # Step 3: Expand Location boundary to adjacent localities or whole city
        candidates = apply_filters(filters, df, relaxed_rating=True, relaxed_budget=True, relaxed_location=True)
        return candidates, "fully_relaxed"
    ```
  * Inform the user clearly in the UI: *"No exact matches found! We've broadened your budget and rating limits to show these options."*

#### 🟡 Scenario D: Text Typos or Alternate Cuisines/Localities
* **Problem**: A user searches for "Chineese" instead of "Chinese", or "Cannaught Place" instead of "Connaught Place".
* **Mitigation Strategy**:
  * Normalize searching strings with custom keyword mappings and string token intersections.
  * Use Python's `difflib` or `rapidfuzz` for location fuzzy matching to map the input query to the closest valid locality in the dataset before running hard Pandas filters.

---

### 3. LLM Orchestration Layer

#### 🔴 Scenario E: Malformed LLM Output / JSON Parsing Failure
* **Problem**: The LLM outputs markdown formatting around its JSON block (e.g. ` ```json ... ``` `) or outputs invalid JSON characters, crashing `json.loads()`.
* **Mitigation Strategy**:
  * Enforce **JSON Mode** where supported by the API (e.g., setting response schema in Gemini or response format in OpenAI).
  * Use a robust regex utility to strip leading/trailing non-JSON characters:
    ```python
    import re
    import json

    def clean_and_parse_json(raw_text):
        try:
            # Extract JSON block using regex if wrapped in markdown code blocks
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # Fallback to manual repair or standard default dictionary
            return get_fallback_recommendations(raw_text)
    ```

#### 🔴 Scenario F: LLM Hallucinations (Fabricating Restaurants)
* **Problem**: The LLM recommends a restaurant that was *not* provided in the candidate list, inventing names or review scores.
* **Mitigation Strategy**:
  * Set `temperature` to `0.1` or `0.2` to enforce strict fidelity.
  * **Id Verification Post-Check**: The orchestrator must match the recommended restaurant IDs in the LLM's response against the candidate list. Any recommended item whose ID is not present in the original candidate dataframe is discarded automatically from the UI to protect data integrity.

#### 🟡 Scenario G: API Rate Limits (HTTP 429) & Token Window Bloat
* **Problem**: Multiple users trigger queries simultaneously, causing the model provider to rate-limit calls.
* **Mitigation Strategy**:
  * **Candidate Cap**: Always limit candidate payloads sent to the LLM to a maximum of 15 restaurants.
  * **Exponential Backoff**: Implement backoff logic utilizing python libraries like `tenacity` to retry API requests if rate limits are hit:
    ```python
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def call_llm_api_with_retry(prompt):
        return llm_client.generate(prompt)
    ```

---

### 4. UI/UX Layer

#### 🟢 Scenario H: Empty Inputs or Extremely Vague Searches
* **Problem**: The user clicks search without entering any keywords or leaves location blank.
* **Mitigation Strategy**:
  * Pre-populate fields with elegant defaults (e.g., Location: "Delhi", Cuisines: "North Indian", Budget: "Medium", Keyword: "Trending").
  * Disable the "Get Recommendations" button in Streamlit unless a valid location is specified.
