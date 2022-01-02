from os import PathLike

import pytest

from coconutools import COCO
from tests.fixtures import Fixtures


class TestDataset:
    @pytest.mark.parametrize(
        "dataset_path,annotation_count,image_count,category_count",
        [(Fixtures.food_nutritions.value, 6, 6, 5)],
    )
    def test_annotation_loading(
            self, dataset_path: PathLike, annotation_count: int, image_count: int, category_count: int
    ) -> None:
        dataset = COCO(annotation_file=dataset_path)

        assert annotation_count == len(dataset.annotations)
        assert image_count == len(dataset.images)
        assert category_count == len(dataset.categories)

    def test_dataset_repr(self):
        dataset = COCO(annotation_file=Fixtures.food_nutritions.value)

        assert "COCO(Food Nutrition Values Dataset)" == str(dataset)
