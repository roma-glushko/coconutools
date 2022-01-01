import json
from os import PathLike
from typing import Optional

"""
class Annotation(TypedDict):
    id: int
    image_id: int
    category_id: int
    segmentation: Sequence[Sequence[float]]
    bbox: Sequence[float]
    ignore: bool
    iscrowd: bool
    area: float


class Image(TypedDict):
    id: int
    width: int
    height: int
    file_name: str
"""

class COCO:
    def __init__(self, annotation_file: PathLike, image_dir: Optional[PathLike] = None) -> None:
        self.annotation_file = annotation_file
        self.image_dir = image_dir

        self.dataset_loaded: bool = False

    @property
    def annotations(self):
        pass

    @property
    def images(self):
        pass

    @property
    def categories(self):
        pass

    def _load_dataset(self) -> None:
        with open(self.annotation_file, "r") as f:
            annotation_file: dict = json.load(f)

        self.images = {
            image["id"]: image
            for image in annotation_file.get("images", [])
        }

        self.categories = {
            category["id"]: category
            for category in annotation_file.get("categories", [])
        }

        self.annotations: Sequence[Annotation] = annotation_file.get("annotations", [])

    def __repr__(self) -> str:
        pass