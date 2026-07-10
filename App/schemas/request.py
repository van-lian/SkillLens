from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    cv_text: str = Field(..., min_length=1, description="Raw CV / resume text")
    role_name: str = Field(..., min_length=1, description="Target job title, e.g. 'android developer'")
    top_n: int = Field(10, ge=1, le=50, description="How many top role skills to compare against")

    class Config:
        json_schema_extra = {
            "example": {
                "cv_text": "Experienced backend developer with 4 years building REST APIs "
                            "in Python and Node.js. Strong SQL and PostgreSQL skills, some "
                            "exposure to Docker and Git-based workflows.",
                "role_name": "android developer",
                "top_n": 10,
            }
        }
