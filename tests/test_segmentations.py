from coconutools import Annotation
from tests.fixtures import SegmentationFormats, generate_annotation_dict


class TestSegmentations:
    def test_bbox_loading(self) -> None:
        annotation_dict = generate_annotation_dict()
        annotation: Annotation = Annotation(**annotation_dict)

        assert annotation.bbox.x == 275
        assert annotation.bbox.y == 207
        assert annotation.bbox.width == 153
        assert annotation.bbox.height == 148

    def test_polygon_segmentation_loading(self) -> None:
        annotation_dict = generate_annotation_dict(
            segmentation_format=SegmentationFormats.polygon
        )

        annotation: Annotation = Annotation(**annotation_dict)
        annotation.segmentation
