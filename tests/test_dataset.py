from os import PathLike
from pathlib import Path

import pytest

from coconutools import ObjectDetectionDataset
from coconutools.exceptions import DatasetCorrupted
from tests.fixtures import Fixtures


class TestDataset:
    @pytest.mark.parametrize(
        "dataset_path,annotation_count,image_count,category_count",
        [(Fixtures.food_nutritions.value, 6, 6, 5)],
    )
    def test_load_annotation(
        self,
        dataset_path: PathLike,
        annotation_count: int,
        image_count: int,
        category_count: int,
    ) -> None:
        dataset: ObjectDetectionDataset = ObjectDetectionDataset(
            annotation_file=dataset_path
        )

        assert annotation_count == len(list(dataset.annotations))
        assert image_count == len(dataset.images)
        assert category_count == len(dataset.categories)

    def test_load_corrupted_annotation(self):
        with pytest.raises(DatasetCorrupted):
            ObjectDetectionDataset(annotation_file=Path(Fixtures.corrupted_annotation))

    def test_dataset_repr(self):
        dataset: ObjectDetectionDataset = ObjectDetectionDataset(
            annotation_file=Fixtures.food_nutritions.value
        )

        assert "COCO('Food Nutrition Values Dataset' v1.0 [Label Studio])" == str(
            dataset
        )
