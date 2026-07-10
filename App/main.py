import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analyze import router as analyze_router
from app.api.roadmap import router as roadmap_router
from app.models.predictor import predictor
from app.services.role_database import role_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model and role database once, at process startup, rather
    # than per-request — DistilBERT is too slow to re-load every call.
    logger.info("Starting up — loading model and role database...")
    predictor.load()
    role_database.load()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Skill Gap Analyzer",
    description="Extracts skills from CVs and job postings, and finds skill gaps against target roles.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before deploying anywhere real
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router, tags=["analyze"])
app.include_router(roadmap_router, tags=["roadmap"])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": predictor.is_ready(),
        "role_db_loaded": role_database.is_ready(),
    }
