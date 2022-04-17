from __future__ import annotations

import warnings
from typing import Any, Final, Generator

from pydantic import BaseModel

from coconutools import Image
from coconutools.annotations.base import AnnotationTypes
from coconutools.dataset import BaseCOCO, RawDataset
from coconutools.exceptions import DatasetNotReferenced


class ImageCaptionAnnotation(BaseModel):
    """
    Image Caption Annotations
    Reference:
    - https://cocodataset.org/#format-data
    """

    id: int
    image_id: int
    capture: str

    def __init__(self, dataset: ImageCaptionDataset | None = None, **data: Any) -> None:
        super().__init__(**data)

        self._dataset = dataset

    @property
    def image(self) -> Image:
        # TODO: move this behavior to the base annotation class
        if not self._dataset:
            raise DatasetNotReferenced(
                "Current annotation has been created outside of any COCO dataset. "
                "Extended image information is not available"
            )

        return self._dataset._get_image(self.image_id)


class ImageCaptionDataset(BaseCOCO):
    type: Final[AnnotationTypes] = AnnotationTypes.IMAGE_CAPTIONING

    def _load_dataset(self, annotation_file: RawDataset) -> None:
        super()._load_dataset(annotation_file)

    @property
    def annotations(self) -> Generator[ImageCaptionAnnotation, None, None]:
        for annotation_info in self._annotations:
            try:
                yield ImageCaptionAnnotation(**annotation_info, dataset=self)
            except TypeError as e:
                warnings.warn(f"Error during annotations parsing: {str(e)}")

    @classmethod
    def can_load(cls, dataset: RawDataset) -> bool:
        annotations: list[dict[str, Any]] = dataset.get("annotations", [])

        if not annotations:
            raise Exception()

        sample_annotation: dict[str, Any] = annotations[0]
        annotation_keys: set[str] = set(sample_annotation.keys())

        return {"id", "image_id", "caption"} == annotation_keys
