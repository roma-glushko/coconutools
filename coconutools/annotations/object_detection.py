from __future__ import annotations

import warnings
from dataclasses import dataclass
from os import PathLike
from typing import Any, Dict, Final, Generator, List, Tuple

from pydantic import BaseModel, Field

from coconutools.annotations.base import AnnotationTypes
from coconutools.dataset import BaseCOCO, RawDataset
from coconutools.exceptions import DatasetNotReferenced
from coconutools.images import Image
from coconutools.segmentations import CompressedRLE_T, PolygonT, UncompressedRLE_T

BBoxT = Tuple[float, float, float, float]


@dataclass
class BBox:
    x: float
    y: float
    width: float
    height: float


class ObjDetCategory(BaseModel):
    """
    Image Category for Object Detection Annotation
    """

    __slots__ = ("id", "name", "supercategory")

    id: int
    name: str
    supercategory: str | None


class ObjDetAnnotation(BaseModel):
    """
    Object Detection Annotation
    """

    __slots__ = (
        "id",
        "image_id",
        "category_id",
        "segmentation",
        "bbox",
        "area",
        "extra",
    )

    id: int
    image_id: int
    category_id: int

    iscrowd: bool = Field(
        ...,
        description="used to label large groups of objects (e.g. iscrowd=1) or a single one (e.g. iscrowd=0)",
    )

    segmentation: List[PolygonT] | UncompressedRLE_T | CompressedRLE_T
    bbox: BBox
    area: float

    extra: Dict[str, Any]  # any custom nodes go here

    def __init__(
        self,
        id: int,
        image_id: int,
        category_id: int,
        iscrowd: bool,
        segmentation: List[PolygonT] | UncompressedRLE_T | CompressedRLE_T,
        bbox: BBoxT,
        area: float,
        dataset: ObjectDetectionDataset | None = None,
        **extra: Dict[str, Any] | None,
    ) -> None:

        super().__init__(**extra)

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
    def category(self) -> ObjDetCategory:
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


class ObjectDetectionDataset(BaseCOCO):
    type: Final[AnnotationTypes] = AnnotationTypes.OBJECT_DETECTION

    def __init__(
        self, annotation_file: PathLike, image_dir: PathLike | None = None
    ) -> None:
        self._categories: list[ObjDetCategory] = []
        self.__category_index: dict[int, ObjDetCategory] = {}

        super().__init__(annotation_file, image_dir)

    def _load_dataset(self, annotation_file: RawDataset) -> None:
        super()._load_dataset(annotation_file)

        self._load_categories(annotation_file.get("categories", []))

    @property
    def annotations(self) -> Generator[ObjDetAnnotation, None, None]:
        for annotation_info in self._annotations:
            try:
                yield ObjDetAnnotation(**annotation_info, dataset=self)
            except TypeError as e:
                warnings.warn(f"Error during annotations parsing: {str(e)}")

    @property
    def categories(self) -> list[ObjDetCategory]:
        return self._categories

    @classmethod
    def can_load(cls, dataset: RawDataset) -> bool:
        annotations: list[dict[str, Any]] = dataset.get("annotations", [])

        if not annotations:
            raise Exception()

        sample_annotation: dict[str, Any] = annotations[0]
        annotation_keys: set[str] = set(sample_annotation.keys())

        return {
            "id",
            "image_id",
            "category_id",
            "segmentation",
            "area",
            "bbox",
            "iscrowd",
        } == annotation_keys

    def _load_categories(self, categories: list[dict[str, Any]]) -> None:
        for category_info in categories:
            category = ObjDetCategory(**category_info)

            self._categories.append(category)
            self.__category_index[category.id] = category

    def _get_category(self, category_id: int) -> ObjDetCategory:
        return self.__category_index[category_id]
