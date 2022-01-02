import json
import warnings
from dataclasses import InitVar, dataclass
from datetime import datetime
from os import PathLike
from typing import Dict, List, Optional, Union

SegmentT = List[float]


@dataclass
class Info:
    year: Optional[int]
    version: Optional[str]
    description: Optional[str]
    contributor: Optional[str]
    url: Optional[str]
    date_created: Optional[datetime]


@dataclass
class Category:
    id: int
    name: str


@dataclass
class Image:
    id: int
    file_name: str
    width: int
    height: int


@dataclass
class Annotation:
    id: int
    image_id: int
    category_id: int
    segmentation: List[SegmentT]
    bbox: List[float]
    ignore: bool
    iscrowd: bool
    area: float

    dataset: InitVar["COCO"] = None

    def __post_init__(self, dataset: "COCO") -> None:
        self._dataset: "COCO" = dataset

    @property
    def image(self) -> Image:
        return self._dataset.get_image(self.image_id)

    @property
    def category(self) -> Category:
        return self._dataset.get_category(self.category_id)


ItemT = Union[Image, Category, Annotation]


class COCO:
    """
    COCO Dataset
    """

    __image_index: Dict[int, Image] = {}
    __category_index: Dict[int, Category] = {}
    __annotation_index: Dict[int, Annotation] = {}

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
    def categories(self) -> List[Category]:
        return self._categories

    def set_image(self, image: Image) -> None:
        self.__image_index[image.id] = image

    def get_image(self, image_id: int) -> Image:
        return self.__image_index[image_id]

    def set_category(self, category: Category) -> None:
        self.__category_index[category.id] = category

    def get_category(self, category_id: int) -> Category:
        return self.__category_index[category_id]

    def set_annotation(self, annotation: Annotation) -> None:
        self.__annotation_index[annotation.id] = annotation

    def get_annotation(self, annotation_id: int) -> Annotation:
        return self.__annotation_index[annotation_id]

    def _load_dataset(self) -> None:
        """ """

        with open(self.annotation_file, "r") as f:
            annotation_file: dict = json.load(f)

        images: List[Image] = []
        categories: List[Category] = []
        annotations: List[Annotation] = []

        for image_info in annotation_file.get("images", []):
            image: Image = Image(**image_info)

            images.append(image)
            self.set_image(image)

        for category_info in annotation_file.get("categories", []):
            category: Category = Category(**category_info)

            categories.append(category)
            self.set_category(category)

        for annotation_info in annotation_file.get("annotations", []):
            try:
                annotation: Annotation = Annotation(**annotation_info, dataset=self)

                annotations.append(annotation)
                self.set_annotation(annotation)
            except TypeError as e:
                warnings.warn(f"Error during annotations parsing: {str(e)}")

        self._info: Info = Info(**annotation_file.get("info", {}))
        self._images: List[Image] = images
        self._categories: List[Category] = categories
        self._annotations: List[Annotation] = annotations

    def __repr__(self) -> str:
        info = self.info

        return f"COCO({info.description})"
