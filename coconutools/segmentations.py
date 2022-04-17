from __future__ import annotations

from typing import List, Tuple, TypedDict

import pycocotools.mask as maskUtils


class UncompressedRLE_T(TypedDict):
    count: List[float]
    size: Tuple[float, float]


CompressedRLE_T = str
PolygonT = List[float]


class Segmentation:
    def mask(self, width: int, height: int):
        rle = self.rle(width, height)

        return maskUtils.decode(rle)

    def rle(self, width: int, height: int):
        raise NotImplementedError()


class PolygonSegmentation(Segmentation):
    def __init__(self, polygons: List[PolygonT]) -> None:
        self.polygons = polygons

    def rle(self, width: int, height: int):
        rles = maskUtils.frPyObjects(self.polygons, height, width)

        return maskUtils.merge(rles)


class UncompressedRLESegmentation(Segmentation):
    def __init__(self, uncompressed_rle: UncompressedRLE_T) -> None:
        self.uncompressed_rle = uncompressed_rle

    def rle(self, width: int, height: int):
        return maskUtils.frPyObjects(self.uncompressed_rle, height, width)


class RLESegmentation(Segmentation):
    def __init__(self, rle: str) -> None:
        self._rle = rle

    def rle(self, width: int, height: int):
        return self._rle


def create_segmentation(
    raw_segmentation: PolygonT | CompressedRLE_T | UncompressedRLE_T,
) -> Segmentation:
    if type(raw_segmentation) == list and len(raw_segmentation[0]) > 4:
        return PolygonSegmentation(polygons=raw_segmentation)

    if (
        type(raw_segmentation) == list
        and type(raw_segmentation[0]) == dict
        and "counts" in raw_segmentation[0]
        and "size" in raw_segmentation[0]
    ):
        return UncompressedRLESegmentation(uncompressed_rle=raw_segmentation)

    return RLESegmentation(
        rle=raw_segmentation
    )  # TODO: Check if this is really not a RLE formatted string
