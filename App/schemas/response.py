from typing import List

from pydantic import BaseModel


class Skill(BaseModel):
    skill: str
    type: str


class AnalyzeResponse(BaseModel):
    role: str
    found: bool
    match_score: int
    have: List[str]
    missing: List[str]
    cv_skills: List[Skill]
    recommendations: List[str]
