from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, TypedDict, Union

from coconutools.exceptions import DatasetNotReferenced
from coconutools.images import Category, Image

if TYPE_CHECKING:
    from coconutools.dataset import COCO


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


@dataclass(init=False)
class Annotation:
    __slots__ = (
        "id",
        "image_id",
        "category_id",
        "segmentation",
        "bbox",
        "iscrowd",
        "area",
        "_dataset",
    )

    id: int
    image_id: int
    category_id: int

    iscrowd: bool

    segmentation: Union[List[PoligonT], UncompressedRLE_T, CompressedRLE_T]
    bbox: BBox
    area: float

    _dataset: Optional["COCO"]

    def __init__(
        self,
        id: int,
        image_id: int,
        category_id: int,
        iscrowd: bool,
        segmentation: Union[List[PoligonT], UncompressedRLE_T, CompressedRLE_T],
        bbox: BBoxT,
        area: float,
        dataset: Optional["COCO"] = None,
    ) -> None:
        self._dataset = dataset

        self.id = id
        self.image_id = image_id
        self.category_id = category_id
        self.iscrowd = iscrowd
        self.area = area

        self.bbox = BBox(*bbox)
        self.segmentation = segmentation

    @property
    def image(self) -> Image:
        if not self._dataset:
            raise DatasetNotReferenced(
                "Current annotation has been created outside of any COCO dataset. "
                "Extended image information is not available"
            )

        return self._dataset._get_image(self.image_id)

    @property
    def category(self) -> Category:
        if not self._dataset:
            raise DatasetNotReferenced(
                "Current annotation has been created outside of any COCO dataset. "
                "Extended category information is not available"
            )

        return self._dataset._get_category(self.category_id)
