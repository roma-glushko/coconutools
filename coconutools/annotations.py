from dataclasses import dataclass
from typing import List, Tuple, TypedDict, Union

from coconutools.dataset import COCO
from coconutools.images import Category, Image


class UncompressedRLE_T(TypedDict):
    count: List[float]
    size: Tuple[float, float]


CompressedRLE_T = str
PoligonT = List[float]
BBoxT = Tuple[float, float, float, float]


@dataclass
class BBox:
    x: float
    y: float
    width: float
    height: float


@dataclass
class Annotation:
    __slots__ = (
        "id",
        "image_id",
        "category_id",
        "segmentation",
        "bbox",
        "iscrowd",
        "area",
    )

    id: int
    image_id: int
    category_id: int

    iscrowd: bool

    segmentation: Union[List[PoligonT], UncompressedRLE_T, CompressedRLE_T]
    bbox: BBox
    area: float

    def __init__(
        self,
        dataset: "COCO",
        id: int,
        image_id: int,
        category_id: int,
        iscrowd: bool,
        segmentation: Union[List[PoligonT], UncompressedRLE_T, CompressedRLE_T],
        bbox: BBoxT,
        area: float,
    ) -> None:
        self._dataset: "COCO" = dataset

        self.id = id
        self.image_id = image_id
        self.category_id = category_id
        self.iscrowd = iscrowd
        self.area = area

        self.bbox = BBox(*bbox)
        self.segmentation = segmentation

    @property
    def image(self) -> Image:
        return self._dataset._get_image(self.image_id)

    @property
    def category(self) -> Category:
        return self._dataset._get_category(self.category_id)
