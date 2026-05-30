# Walkthrough Guide

This guide walks through running and verifying the **Zomato AI Restaurant Recommender** end to end.

## Prerequisites

1. Python 3.10+
2. Virtual environment with dependencies installed:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and configure:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

## Step 1 — Load & cache the dataset

```bash
python -m src.data_loader
```

Verify `data/zomato_cleaned.parquet` exists with columns: `name`, `locality`, `city`, `cuisines`, `average_cost_for_two`, `aggregate_rating`, `votes`.

## Step 2 — Test the retrieval engine

```bash
python -m src.retriever
```

Try an over-restricted query in code or via the app (e.g. French cuisine, Low budget, rating 4.9). The system should relax filters and show a message like:

> We relaxed your rating constraint slightly (now >= 3.5) to find matching locations!

## Step 3 — Test the LLM connector

```bash
python -m src.llm_connector
```

Or run the live Groq smoke test:

```bash
python scripts/run_live_llm_test.py
```

## Step 4 — Launch the Streamlit UI

```bash
streamlit run src/app.py
```

1. Filter a **Location** using the search box + dropdown
2. Pick **Cuisines**, **Budget**, and **Minimum Rating**
3. Enter a qualitative mood (e.g. *"rooftop date night with cocktails"*)
4. Click **Find Restaurants**

You should see:
- A spinner during retrieval
- A skeleton loader while the LLM ranks candidates
- Styled cards with rating badges, cost pills, and **AI Insight** reasoning
- A relaxation banner if filters were broadened

## Step 5 — Run automated tests

```bash
pytest tests/ -v
```

Phase 5 verification (requires cached parquet):

```bash
pytest tests/test_phase5.py -v
```

Key scenario: `test_over_restricted_query_does_not_crash` — rating 4.9 + Low budget + niche cuisine.

## Fallback behaviour (Phase 5)

When strict filters return too few matches, the retriever relaxes in order:

1. **Budget** — removes budget tier constraint
2. **Rating** — lowers minimum rating by 0.5 (floor 3.0)
3. **Nearby localities** — expands to popular areas in the same city
4. **Location-only** — drops cuisine/budget constraints
5. **Global** — top-rated restaurants overall

The UI displays a human-readable explanation for each relaxation step.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Offline mode banner | Set `GROQ_API_KEY` in `.env` and save the file |
| Empty location list | Run `python -m src.data_loader` to generate parquet |
| Groq model error | Try `GROQ_MODEL=llama-3.3-70b-versatile` in `.env |
| No results after search | Pick a broader location or lower the rating slider |
