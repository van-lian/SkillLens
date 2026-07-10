import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

ROLE_DB_PATH = Path(__file__).resolve().parents[2] / "model" / "role_database.json"


class RoleDatabase:
    """Wraps the role_database.json produced by build_role_database() in the
    training notebook: {role_title: [{"skill", "type", "freq"}, ...]}.
    """

    def __init__(self, path: Path = ROLE_DB_PATH):
        self.path = path
        self._db: dict = {}

    def load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Role database not found: {self.path}\n"
                "Generate it with the training notebook (Stage 3) or "
                "scripts/build_role_database.py and place it at model/role_database.json."
            )
        self._db = json.loads(self.path.read_text())
        logger.info("Loaded role database with %d roles", len(self._db))

    def is_ready(self) -> bool:
        return bool(self._db)

    def list_roles(self) -> List[str]:
        return sorted(self._db.keys())

    def get(self, role_name: str) -> list:
        return self._db.get(role_name, [])

    def find_closest(self, role_name: str) -> Optional[str]:
        """Case-insensitive exact match on the cleaned title. Good enough for
        now; swap in fuzzy matching (e.g. rapidfuzz) later if titles don't
        line up well enough in practice.
        """
        target = role_name.strip().lower()
        for role in self._db:
            if role.lower() == target:
                return role
        return None


# Single shared instance, loaded once in main.py's lifespan handler.
role_database = RoleDatabase()
