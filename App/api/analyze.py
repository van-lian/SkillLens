from fastapi import APIRouter, HTTPException

from app.models.comparator import compare_to_role, recommend
from app.models.extractor import extract_skills
from app.schemas.request import AnalyzeRequest
from app.schemas.response import AnalyzeResponse

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    """Extract skills from a CV and compare them against a target role's
    top required skills, producing a match score and recommendations.
    """
    try:
        cv_skills = extract_skills(payload.cv_text)
        gap = compare_to_role(payload.cv_text, payload.role_name, top_n=payload.top_n)
        recs = recommend(gap)
    except RuntimeError as e:
        # Model not loaded yet — shouldn't happen once startup completes,
        # but fail loudly instead of a bare 500 if it does.
        raise HTTPException(status_code=503, detail=str(e))

    return AnalyzeResponse(
        role=gap["role"],
        found=gap["found"],
        match_score=gap["match_score"],
        have=gap["have"],
        missing=gap["missing"],
        cv_skills=cv_skills,
        recommendations=recs,
    )
