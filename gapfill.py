import json, re

GAP_PROMPT = """You are a coverage checker.
Given the JSON summary below, list unanswered facts
that a reader would still want.  Reply with 10 bullet
points max of good efficient google queries that would answer the questions, no commentary, if you find any events, look up more details about them, also linkeding contacts in marketing, cse, hr, office, remember we are in the year 2025, maybe match event names with company cities and countries... be smart about it, don't clump too much into one query, and don't repeat yourself."""

BULLET_RE = re.compile(r"[-•]\s*(.+)")

def _extract_bullets(text: str) -> list[str]:
    return BULLET_RE.findall(text)

def gap_fill_once(
    client,                        # OpenAI client
    merged_summary_json: dict,
    search_fn,                     # callable(company_name, structured=True)
    company_name: str,
) -> list[dict]:
    """
    Ask GPT-4o-mini for coverage gaps, turn them into extra Google queries,
    run one more search cycle, and return NEW structured items.
    """
    gaps_txt = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": GAP_PROMPT},
            {"role": "user",   "content": json.dumps(merged_summary_json)}
        ]
    ).choices[0].message.content
    MAX_GAP_QUERIES = 10

    gap_queries = _extract_bullets(gaps_txt)
    gap_queries = gap_queries[:MAX_GAP_QUERIES] 
    if not gap_queries:
        return []                          # nothing to add

    # Run an extra Google CSE for each gap, concatenate & dedup
    
    extra_items, seen = [], set()
    for q in gap_queries:
        q_str = f"{company_name} {q}"
        print(f"[GAP] searching → {q_str}")
        for it in search_fn(q_str, structured=True, num_results=5):  # ↓ limit hits
            if it["link"] in seen:
                continue
            seen.add(it["link"])
            extra_items.append(it)
        print(f"[GAP] {q} done • new items: {len(extra_items)}")
    return extra_items
