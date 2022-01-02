from os import PathLike
from typing import Any, Dict

import pytest
from pytest import approx

from coconutools import COCO, Annotation, Category, Image
from tests.fixtures import Fixtures


class TestAnnotations:
    @pytest.mark.parametrize(
        "dataset_path, expected_annotation",
        [
            (
                Fixtures.food_nutritions.value,
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 3,
                    "ignore": 0,
                    "iscrowd": 0,
                    "area": 3283691.2497,
                },
            )
        ],
    )
    def test_annotations_loaded(
        self,
        dataset_path: PathLike,
        expected_annotation: Dict[str, Any],
    ) -> None:
        dataset: COCO = COCO(annotation_file=dataset_path)

        annotation = dataset.annotations[0]

        assert expected_annotation.get("id") == annotation.id
        assert expected_annotation.get("image_id") == annotation.image_id
        assert expected_annotation.get("category_id") == annotation.category_id
        assert expected_annotation.get("ignore") == annotation.ignore
        assert expected_annotation.get("iscrowd") == annotation.iscrowd
        assert expected_annotation.get("area") == approx(annotation.area, 2)

    def test_category_reference(self):
        dataset: COCO = COCO(annotation_file=Fixtures.food_nutritions.value)

        annotation: Annotation = dataset.annotations[0]
        category: Category = annotation.category

        assert annotation.category_id == category.id
        assert category.name == "Nutritions"

    def test_image_reference(self):
        dataset: COCO = COCO(annotation_file=Fixtures.food_nutritions.value)

        annotation: Annotation = dataset.annotations[0]
        image: Image = annotation.image

        assert annotation.image_id == image.id
        assert image.file_name == "images/1/96021e5b-IMG_2055.jpeg"
        assert image.width == 3024
        assert image.height == 4032