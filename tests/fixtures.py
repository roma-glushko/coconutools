from enum import Enum
from os.path import dirname
from pathlib import Path

FIXTURE_DIR: Path = Path(dirname(__file__) + "/fixtures").resolve()


class Fixtures(str, Enum):
    food_nutritions = FIXTURE_DIR / "food_nutritions.json"
