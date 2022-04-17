from os import PathLike

import pytest

from coconutools import Info, ObjectDetectionDataset
from tests.fixtures import Fixtures


class TestDatasetInfoLoading:
    @pytest.mark.parametrize(
        "dataset_path, description, year, version, url, contributor",
        [
            (
                Fixtures.food_nutritions.value,
                "Food Nutrition Values Dataset",
                2021,
                "1.0",
                "https://food-nutrition-dataset.com",
                "Label Studio",
            )
        ],
    )
    def test_info_metadata(
        self,
        dataset_path: PathLike,
        description: str,
        year: int,
        version: str,
        url: str,
        contributor: str,
    ) -> None:
        dataset: ObjectDetectionDataset = ObjectDetectionDataset(
            annotation_file=dataset_path
        )

        dataset_info: Info = dataset.info

        assert description == dataset_info.description
        assert year == dataset_info.year
        assert version == dataset_info.version
        assert url == dataset_info.url
        assert contributor == dataset_info.contributor
