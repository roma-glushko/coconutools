from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from coconutools.exceptions import DatasetNotReferenced

if TYPE_CHECKING:
    from coconutools.dataset import COCO


@dataclass
class License:
    """
    Image Licence
    """

    id: int
    name: str
    url: str


@dataclass(init=False)
class Category:
    """
    Image Category
    """

    __slots__ = ("id", "name", "supercategory")

    id: int
    name: str
    supercategory: Optional[str]

    def __init__(self, id: int, name: str, supercategory: Optional[str] = None) -> None:
        self.id = id
        self.name = name
        self.supercategory = supercategory


@dataclass(init=False)
class Image:
    __slots__ = (
        "id",
        "file_name",
        "width",
        "height",
        "coco_url",
        "flickr_url",
        "date_captured",
        "license_id",
        "_dataset",
    )

    id: int
    file_name: str
    width: int
    height: int
    coco_url: Optional[str]
    flickr_url: Optional[str]
    date_captured: Optional[datetime]
    license_id: Optional[int]

    def __init__(
        self,
        id: int,
        file_name: str,
        width: int,
        height: int,
        coco_url: Optional[str] = None,
        flickr_url: Optional[str] = None,
        date_captured: Optional[datetime] = None,
        license: Optional[int] = None,
        dataset: Optional["COCO"] = None,
    ):
        self._dataset: "COCO" = dataset

        self.id = id
        self.file_name = file_name
        self.width = width
        self.height = height
        self.coco_url = coco_url
        self.flickr_url = flickr_url
        self.date_captured = date_captured
        self.license_id = license

    @property
    def license(self) -> Optional[License]:
        if not self.license_id:
            return None

        if not self._dataset:
            raise DatasetNotReferenced(
                f"The image (ID={self.id}, path={self.file_name}) has been created outside of any COCO dataset. "
                "Extended license information is not available"
            )

        return self._dataset._get_licence(self.license_id)
