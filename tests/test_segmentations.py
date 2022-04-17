from coconutools import ObjDetAnnotation
from tests.fixtures import generate_annotation_dict


class TestSegmentations:
    def test_bbox_loading(self) -> None:
        annotation_dict = generate_annotation_dict()
        annotation: ObjDetAnnotation = ObjDetAnnotation(**annotation_dict)

        assert annotation.bbox.x == 275
        assert annotation.bbox.y == 207
        assert annotation.bbox.width == 153
        assert annotation.bbox.height == 148

    # def test_polygon_segmentation_loading(self) -> None:
    #     dataset = COCO(annotation_file=Path(Fixtures.food_nutritions.value))
    #
    #     annotation: Annotation = dataset.annotations[0]
    # image: Image = annotation.image

    # uncompressed_rle = convert_polygons_to_rle(
    #     annotation.segmentation, image_width=image.width, image_height=image.height
    # )

    # mask = np.maximum(coco.annToMask(uncompressed_rle), mask)
