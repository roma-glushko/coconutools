import json
import warnings
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime
from os import PathLike
from typing import Any, Dict, List, Optional, Union

from coconutools.annotations import Annotation
from coconutools.images import Category, Image, License

with suppress(ModuleNotFoundError):
    import pandas


@dataclass
class Info:
    __slots__ = ("year", "version", "description", "contributor", "url", "date_created")

    year: Optional[int]
    version: Optional[str]
    description: Optional[str]
    contributor: Optional[str]
    url: Optional[str]
    date_created: Optional[datetime]


ItemT = Union[Image, Category, Annotation]


class COCO:
    """
    COCO Dataset

    Description of COCO format: https://cocodataset.org/#format-data
    """

    __image_index: Dict[int, Image] = {}
    __category_index: Dict[int, Category] = {}
    __annotation_index: Dict[int, Annotation] = {}
    __license_index: Dict[int, License] = {}

    def __init__(
        self, annotation_file: PathLike, image_dir: Optional[PathLike] = None
    ) -> None:
        self.annotation_file = annotation_file
        self.image_dir = image_dir

        self._load_dataset()

    @property
    def info(self) -> Info:
        return self._info

    @property
    def annotations(self) -> List[Annotation]:
        return self._annotations

    @property
    def images(self) -> List[Image]:
        return self._images

    @property
    def licences(self) -> List[License]:
        return self._licenses

    @property
    def categories(self) -> List[Category]:
        return self._categories

    def _set_image(self, image: Image) -> None:
        self.__image_index[image.id] = image

    def _get_image(self, image_id: int) -> Image:
        return self.__image_index[image_id]

    def _set_category(self, category: Category) -> None:
        self.__category_index[category.id] = category

    def _get_category(self, category_id: int) -> Category:
        return self.__category_index[category_id]

    def _set_licence(self, license: License) -> None:
        self.__license_index[license.id] = license

    def _get_licence(self, licence_id: int) -> License:
        return self.__license_index[licence_id]

    def _set_annotation(self, annotation: Annotation) -> None:
        self.__annotation_index[annotation.id] = annotation

    def _get_annotation(self, annotation_id: int) -> Annotation:
        return self.__annotation_index[annotation_id]

    def _load_dataset(self) -> None:
        """
        Loads a COCO annotation JSON file
        """

        with open(self.annotation_file, "r") as f:
            annotation_file: dict = json.load(f)

        assert type(annotation_file) == dict

        self._images: List[Image] = []
        self._categories: List[Category] = []
        self._licenses: List[License] = []
        self._annotations: List[Annotation] = []

        self._info: Info = Info(**annotation_file.get("info", {}))

        for category_info in annotation_file.get("categories", []):
            category: Category = Category(**category_info)

            self._categories.append(category)
            self._set_category(category)

        for license_info in annotation_file.get("licenses", []):
            licence: License = License(**license_info)

            self._licenses.append(licence)
            self._set_licence(licence)

        for image_info in annotation_file.get("images", []):
            image: Image = Image(**image_info)

            self._images.append(image)
            self._set_image(image)

        for annotation_info in annotation_file.get("annotations", []):
            try:
                annotation: Annotation = Annotation(**annotation_info, dataset=self)

                self._annotations.append(annotation)
                self._set_annotation(annotation)
            except TypeError as e:
                warnings.warn(f"Error during annotations parsing: {str(e)}")

    def df(self) -> "pandas.DataFrame":
        """
        Convert COCO dataset to pandas.DataFrame

        :return: pandas.DataFrame
        """
        try:
            import pandas as pd

            data: List[Dict[str, Any]] = [
                {
                    **asdict(annotation),
                    "category_name": annotation.category.name,
                    "image_path": annotation.image.file_name,
                    "image_width": annotation.image.width,
                    "image_height": annotation.image.height,
                }
                for annotation in self.annotations
            ]

            return pd.DataFrame(data)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "In order to be able to convert your COCO dataset to DataFrame you need to "
                "have pandas installed in your project: "
                "- pip install pandas"
                "- poetry add pandas"
            )

    def __repr__(self) -> str:
        info = self.info

        return f"COCO({info.description})"
