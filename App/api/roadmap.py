from fastapi import APIRouter

router = APIRouter()


@router.post("/roadmap")
def roadmap():
    """Placeholder for the AI-generated learning roadmap feature.

    Will call app/models/roadmap_generator.py once that's built, using
    app/services/prompt_builder.py + resource_mapper.py to turn a
    compare_to_role() result into a step-by-step learning plan.
    """
    return {"detail": "Not implemented yet."}
