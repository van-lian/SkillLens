import os
import json
import time
from collections import Counter

import pandas as pd
from groq import Groq



# ---------------------------------------------------------------------------
# JSON repair — handles truncated or malformed model output
# ---------------------------------------------------------------------------

def repair_json(text: str) -> list:
    """Try progressively more lenient parsing strategies."""
    import re

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract the JSON array portion
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3: extract individual row objects and parse each one
    results = []
    for obj_match in re.finditer(r'\{[^{}]*"row"\s*:\s*(\d+)[^{}]*\}', text, re.DOTALL):
        try:
            obj = json.loads(obj_match.group())
            results.append(obj)
        except json.JSONDecodeError:
            pass
    if results:
        return results

    # Strategy 4: manually extract row + skills pairs
    results = []
    for block in re.finditer(r'"row"\s*:\s*(\d+).*?"skills"\s*:\s*(\[.*?\])', text, re.DOTALL):
        try:
            row_num = int(block.group(1))
            skills  = json.loads(block.group(2))
            results.append({"row": row_num, "skills": skills})
        except Exception:
            pass
    return results

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
API_KEY    = "a"   # from https://console.groq.com
MODEL      = "llama-3.3-70b-versatile"     # fast + free on Groq
INPUT_CSV  = "dataset/cleaned_data/dataset.csv"
OUTPUT_CSV = "dataset/cleaned_data/skill_mapping_output.csv"
TEXT_COL   = "cleaned_text"
ROLE_COL   = "title"
ROW_LIMIT  = 10_000
BATCH_SIZE = 10      # rows per API call
SAVE_EVERY = 100     # save progress every N rows

# Groq free tier: 30 RPM, 14400 req/day
# 10 rows/batch → 1000 batches for 10k rows
# Sleep 3s between calls → ~20 RPM, safely under 30 RPM
# Total time: ~50 minutes
SLEEP_SEC  = 3


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(rows: list) -> str:
    numbered = "\n\n".join(f"Row {i+1}: {str(text)[:400]}" for i, text in enumerate(rows))
    return f"""You are a skill extraction expert. For each row below extract ONLY specific, actionable skills, tools, technologies, and competencies.

REMOVE generic filler such as: skill, experience, knowledge of, understanding of, ability to, proficiency in, familiar with, years of, strong, excellent, good, proven, demonstrated, required, preferred, and similar vague phrases.

KEEP specific items such as:
- Technical: SQL, Python, Java, machine learning, data analysis, AutoCAD
- Soft: leadership, negotiation, team management, public speaking
- Tools: Excel, Salesforce, Jira, Adobe Photoshop, Tableau, SAP
- Domain: financial modeling, supply chain, clinical research, tax planning
- Certifications: PMP, CPA, AWS Certified, CISSP, Six Sigma

The dataset contains mixed roles — not just tech. Extract skills relevant to any profession.

Assign each skill one category: technical | soft | tool | domain | certification

Respond ONLY with a valid JSON array — no markdown fences, no explanation.
Format: [{{"row": <number>, "skills": [{{"name": "<skill>", "category": "<category>"}}]}}]

Rows:
{numbered}"""


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_groq(client, prompt: str, retries: int = 5) -> list:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4096,
            )
            text = response.choices[0].message.content.strip()
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = repair_json(text)
            if parsed:
                return parsed
            print(f"    [warn] could not parse JSON (attempt {attempt+1}), retrying...")
            time.sleep(2)
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = 30 * (attempt + 1)
                print(f"    [rate limit] waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [warn] API error (attempt {attempt+1}): {e}")
                time.sleep(3 * (attempt + 1))
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df)} total rows")

    df = df[[ROLE_COL, TEXT_COL]].head(ROW_LIMIT).copy()
    df.reset_index(drop=True, inplace=True)
    print(f"  Processing {len(df)} rows")

    # Resume from existing output if present
    done_indices = set()
    results_rows = []
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        if "row_index" in existing.columns:
            done_indices = set(existing["row_index"].tolist())
        results_rows = existing.to_dict("records")
        print(f"  Resuming — {len(done_indices)} rows already done")

    total    = len(df)
    texts    = df[TEXT_COL].fillna("").tolist()
    titles   = df[ROLE_COL].fillna("").tolist()
    batches  = [list(range(i, min(i + BATCH_SIZE, total))) for i in range(0, total, BATCH_SIZE)]
    pending_batches = [b for b in batches if not all(idx in done_indices for idx in b)]

    print(f"\n  Total batches   : {len(batches)}")
    print(f"  Remaining       : {len(pending_batches)}")
    print(f"  Batch size      : {BATCH_SIZE} rows")
    print(f"  Sleep between   : {SLEEP_SEC}s")
    est_min = len(pending_batches) * SLEEP_SEC / 60
    print(f"  Est. time       : ~{est_min:.0f} min")
    print(f"  Output          : {OUTPUT_CSV}\n")

    client    = Groq(api_key=API_KEY)
    processed = len(done_indices)

    for bi, batch_indices in enumerate(batches):
        if all(idx in done_indices for idx in batch_indices):
            continue

        pending     = [idx for idx in batch_indices if idx not in done_indices]
        batch_texts = [texts[idx] for idx in pending]

        print(f"  Batch {bi+1}/{len(batches)}  rows {pending[0]+1}–{pending[-1]+1} ...", end=" ", flush=True)
        parsed = call_groq(client, build_prompt(batch_texts))

        skill_map = {}
        for item in parsed:
            local_idx = item["row"] - 1
            if 0 <= local_idx < len(pending):
                skill_map[pending[local_idx]] = item.get("skills", [])

        for idx in pending:
            skills = skill_map.get(idx, [])
            results_rows.append({
                "row_index":            idx,
                ROLE_COL:               titles[idx],
                TEXT_COL:               texts[idx][:200],
                "skills_technical":     "; ".join(s.get("name","") for s in skills if s.get("category") == "technical" and s.get("name")),
                "skills_soft":          "; ".join(s.get("name","") for s in skills if s.get("category") == "soft" and s.get("name")),
                "skills_tool":          "; ".join(s.get("name","") for s in skills if s.get("category") == "tool" and s.get("name")),
                "skills_domain":        "; ".join(s.get("name","") for s in skills if s.get("category") == "domain" and s.get("name")),
                "skills_certification": "; ".join(s.get("name","") for s in skills if s.get("category") == "certification" and s.get("name")),
                "all_skills":           "; ".join(s.get("name","") for s in skills if s.get("name")),
                "skill_count":          len(skills),
            })
            done_indices.add(idx)

        processed += len(pending)
        pct = processed / total * 100
        print(f"✓  ({processed}/{total}  {pct:.1f}%)")

        if processed % SAVE_EVERY < BATCH_SIZE or processed == total:
            pd.DataFrame(results_rows).to_csv(OUTPUT_CSV, index=False)
            print(f"    [saved {len(results_rows)} rows → {OUTPUT_CSV}]")

        time.sleep(SLEEP_SEC)

    # Final save + summary
    out_df = pd.DataFrame(results_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    all_skills = []
    for row in out_df["all_skills"]:
        all_skills.extend(s.strip() for s in str(row).split(";") if s.strip())

    counts = Counter(s.lower() for s in all_skills)
    top10  = counts.most_common(10)

    print("\n" + "=" * 52)
    print("  DONE")
    print("=" * 52)
    print(f"  Rows processed   : {len(out_df)}")
    print(f"  Total extractions: {sum(counts.values())}")
    print(f"  Unique skills    : {len(counts)}")
    print(f"  Avg skills/row   : {out_df['skill_count'].mean():.1f}")
    print(f"  Output saved to  : {OUTPUT_CSV}")
    if top10:
        print("\n  Top 10 skills:")
        for skill, cnt in top10:
            print(f"    {cnt:>5}×  {skill}")
    print("=" * 52)


if __name__ == "__main__":
    main()