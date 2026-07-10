from typing import Dict, List

from app.models.extractor import extract_skills
from app.services.role_database import role_database


def compare_to_role(cv_text: str, role_name: str, top_n: int = 10) -> Dict:
    """Same logic as `compare_to_role` in 06_Data_NLP.ipynb (Stage 5), plus a
    `found` flag so the API can tell "0% match" apart from "unknown role".
    """
    resolved_role = role_database.find_closest(role_name)
    if resolved_role is None:
        return {"role": role_name, "found": False, "have": [], "missing": [], "match_score": 0}

    cv_skill_set = {item["skill"] for item in extract_skills(cv_text)}
    required = [s["skill"] for s in role_database.get(resolved_role)[:top_n]]

    have = [s for s in required if s in cv_skill_set]
    missing = [s for s in required if s not in cv_skill_set]
    match_score = round(100 * len(have) / len(required)) if required else 0

    return {
        "role": resolved_role,
        "found": True,
        "have": have,
        "missing": missing,
        "match_score": match_score,
    }


def recommend(gap_result: Dict) -> List[str]:
    if not gap_result["found"]:
        return [f"Role '{gap_result['role']}' isn't in the role database yet."]
    if not gap_result["missing"]:
        return [f"Great fit — you already cover the top skills for '{gap_result['role']}'."]
    return [f"Consider learning or highlighting: {s}" for s in gap_result["missing"]]
