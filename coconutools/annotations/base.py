from enum import Enum, unique


@unique
class AnnotationTypes(Enum):
    """
    Types of the annotations that COCO dataset can store
    Reference:
    - https://cocodataset.org/#format-data
    """

    OBJECT_DETECTION = "object_detection"  # along with stuff segmentation
    KEYPOINT_DETECTION = "keypoint_detection"
    PANOPTIC_SEGMENTATION = "panoptic_segmentation"
    IMAGE_CAPTIONING = "image_captation"
    DENSE_POSE = "dense_pose_detection"
