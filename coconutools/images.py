from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from coconutools.dataset import BaseCOCO
from coconutools.exceptions import DatasetNotReferenced


class License(BaseModel):
    """
    Image Licence
    """

    __slots__ = (
        "id",
        "name",
        "url",
    )

    id: int
    name: str
    url: str


class Image(BaseModel):
    id: int
    file_name: str
    width: int
    height: int
    coco_url: str | None
    flickr_url: str | None
    date_captured: datetime | None
    license_id: int | None

    _dataset: BaseCOCO

    @property
    def license(self) -> Optional[License]:
        if not self.license_id:
            return None

        if not self._dataset:
            raise DatasetNotReferenced(
                f"The image (ID={self.id}, path={self.file_name}) has been created outside of any COCO dataset. "
                "Extended license information is not available"
            )

        return self._dataset._get_licence(self.license_id)
