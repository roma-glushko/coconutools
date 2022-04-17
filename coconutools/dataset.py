from __future__ import annotations

import json
from datetime import datetime
from json import JSONDecodeError
from os import PathLike
from typing import Any, Optional, TypedDict

from pydantic import BaseModel

from coconutools.exceptions import DatasetCorrupted, DatasetFormatNotValid
from coconutools.images import Image, License


class RawDataset(TypedDict):
    info: dict[str, Any] | None
    licenses: list[dict[str, Any]] | None
    annotations: list[dict[str, Any]]
    images: list[dict[str, str]]
    categories: list[dict[str, Any]] | None


class Info(BaseModel):
    __slots__ = ("year", "version", "description", "contributor", "url", "date_created")

    year: int | None
    version: str | None
    description: str | None
    contributor: str | None
    url: str | None
    date_created: datetime | None


def _load_annotation_file(annotation_path: PathLike) -> RawDataset:
    """
    Loads and validations a COCO annotation JSON file

    :param annotation_file (PathLike): Path to the annotation file
    :return: Content of annotation file
    """
    try:
        annotation_file: RawDataset = json.load(open(annotation_path, "r"))
    except JSONDecodeError as e:
        raise DatasetCorrupted(
            f"COCO dataset {annotation_path} seems to be corrupted or not a valid JSON file"
        ) from e

    assert type(annotation_file) == dict

    dataset_properties = set(annotation_file.keys())

    if not dataset_properties >= {"annotations", "images"}:
        raise DatasetFormatNotValid(
            "COCO dataset should have at least one annotation, image and category"
        )

    return annotation_file


class BaseCOCO:
    """
    COCO Dataset

    Description of COCO format: https://cocodataset.org/#format-data
    """

    def __init__(
        self, annotation_file: PathLike, image_dir: Optional[PathLike] = None
    ) -> None:
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self._annotations: list[dict[str, str]] = []

        self.__image_index: dict[int, Image] = {}
        self.__license_index: dict[int, License] = {}

        raw_dataset: RawDataset = _load_annotation_file(self.annotation_file)

        self._load_dataset(raw_dataset)

    @property
    def info(self) -> Info:
        return self._info

    @property
    def images(self) -> list[Image]:
        return self._images

    @property
    def licences(self) -> list[License]:
        return self._licenses

    def _add_image(self, image: Image) -> None:
        self._images.append(image)

        self.__image_index[image.id] = image

    def _get_image(self, image_id: int) -> Image:
        return self.__image_index[image_id]

    def _add_licence(self, license: License) -> None:
        self._licenses.append(license)

        self.__license_index[license.id] = license

    def _get_licence(self, licence_id: int) -> License:
        return self.__license_index[licence_id]

    def _load_dataset(self, annotation_file: RawDataset) -> None:
        """
        Loads a COCO annotation JSON file
        """

        self._images: list[Image] = []
        self._licenses: list[License] = []

        self._info: Info = Info(**annotation_file.get("info", {}))

        for license_info in annotation_file.get("licenses", []):
            licence: License = License(**license_info)

            self._add_licence(licence)

        for image_info in annotation_file.get("images", []):
            image: Image = Image(**image_info, dataset=self)

            self._add_image(image)

    def __repr__(self) -> str:
        info = self.info

        return (
            f"{self.__class__.__name__}"
            f"('{info.description}' v{info.version} [{info.contributor}], images: {len(self._images)})"
        )
