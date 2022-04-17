from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from typing import Any, Final, Generator

from pydantic import BaseModel, Field

from coconutools.annotations.base import AnnotationTypes
from coconutools.annotations.object_detection import ObjDetAnnotation, ObjDetCategory
from coconutools.dataset import BaseCOCO, RawDataset


class Visibility(int, Enum):
    NOT_LABELED = 0
    LABELED_NOT_VISIBLE = 1
    LABELED_VISIBLE = 2


class Keypoint(BaseModel):
    x: int
    y: int
    visibility: Visibility


@dataclass
class KeypointCategory(ObjDetCategory):
    """
    Keypoint Category
    """

    keypoints: list[str] = Field(..., description="a length k array of keypoint names")
    skeleton: list[int] = Field(
        ..., description="a list of keypoint edge pairs. Used for visualization"
    )


@dataclass
class KeypointAnnotation(ObjDetAnnotation):
    keypoints: list[Keypoint]
    num_keypoints: int


class KeypointDetectionDataset(BaseCOCO):
    type: Final[AnnotationTypes] = AnnotationTypes.KEYPOINT_DETECTION

    def __init__(
        self, annotation_file: PathLike, image_dir: PathLike | None = None
    ) -> None:
        self._categories: list[KeypointCategory] = []
        self.__category_index: dict[int, KeypointCategory] = {}

        super().__init__(annotation_file, image_dir)

    def _load_dataset(self, annotation_file: RawDataset) -> None:
        super()._load_dataset(annotation_file)

        self._load_categories(annotation_file.get("categories") or [])

    @property
    def annotations(self) -> Generator[KeypointAnnotation, None, None]:
        for annotation_info in self._annotations:
            try:
                yield KeypointAnnotation(
                    **annotation_info, dataset=self
                )  # TODO: find solution
            except TypeError as e:
                warnings.warn(f"Error during annotations parsing: {str(e)}")

    @property
    def categories(self) -> list[KeypointCategory]:
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
            category = KeypointCategory(**category_info)

            self._categories.append(category)
            self.__category_index[category.id] = category

    def _get_category(self, category_id: int) -> KeypointCategory:
        return self.__category_index[category_id]
