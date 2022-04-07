from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from coconutools.exceptions import DatasetNotReferenced
from coconutools.images import Category, Image
from coconutools.segmentations import CompressedRLE_T, PolygonT, UncompressedRLE_T

if TYPE_CHECKING:
    from coconutools.dataset import COCO

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
        "extra",
    )

    id: int
    image_id: int
    category_id: int

    iscrowd: bool

    segmentation: Union[List[PolygonT], UncompressedRLE_T, CompressedRLE_T]
    bbox: BBox
    area: float

    extra: Dict[str, Any]  # any custom nodes go here

    _dataset: Optional["COCO"]

    def __init__(
        self,
        id: int,
        image_id: int,
        category_id: int,
        iscrowd: bool,
        segmentation: Union[List[PolygonT], UncompressedRLE_T, CompressedRLE_T],
        bbox: BBoxT,
        area: float,
        dataset: Optional["COCO"] = None,
        **extra: Optional[Dict[str, Any]]
    ) -> None:
        self._dataset = dataset

        self.id = id
        self.image_id = image_id
        self.category_id = category_id
        self.iscrowd = iscrowd
        self.area = area

        self.bbox = BBox(*bbox)
        self.segmentation = segmentation

        self.extra = extra

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

    def mask(self):
        # TODO: convert annotation to the mask
        pass

    def rle(self):
        # convert annotation to RLE
        pass
