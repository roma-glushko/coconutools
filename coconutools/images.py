from dataclasses import InitVar, dataclass
from datetime import datetime
from typing import Optional

from coconutools.dataset import COCO


@dataclass
class License:
    """
    Image Licence
    """

    id: int
    name: str
    url: str


@dataclass
class Category:
    """
    Image Category
    """

    __slots__ = ("id", "name", "supercategory")

    id: int
    name: str
    supercategory: Optional[str]


@dataclass
class Image:
    __slots__ = ("id", "file_name", "width", "height")

    id: int
    file_name: str
    coco_url: Optional[str]
    flickr_url: Optional[str]
    date_captured: Optional[datetime]
    license_id: Optional[int]
    width: int
    height: int

    dataset: InitVar["COCO"] = None

    def __post_init__(self, dataset: "COCO") -> None:
        self._dataset: "COCO" = dataset  # type: ignore

    @property
    def license(self) -> Optional[License]:
        if not self.license_id:
            return None

        return self._dataset._get_licence(self.license_id)
