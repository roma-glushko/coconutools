from os import PathLike

import pytest

from coconutools import COCO
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
        dataset = COCO(annotation_file=dataset_path)

        assert annotation_count == len(dataset.annotations)
        assert image_count == len(dataset.images)
        assert category_count == len(dataset.categories)

    def test_load_corrupted_annotation(self):
        with pytest.raises(DatasetCorrupted):
            COCO(annotation_file=Fixtures.corrupted_annotation)

    def test_dataset_repr(self):
        dataset = COCO(annotation_file=Fixtures.food_nutritions.value)

        assert "COCO(Food Nutrition Values Dataset)" == str(dataset)

    def test_converting_to_dataframe(self):
        dataset = COCO(annotation_file=Fixtures.food_nutritions.value)

        dataframe = dataset.df()

        assert [
            "id",
            "image_id",
            "category_id",
            "segmentation",
            "bbox",
            "ignore",
            "iscrowd",
            "area",
            "category_name",
            "image_path",
            "image_width",
            "image_height",
        ] == list(dataframe.columns)
