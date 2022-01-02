# coconutools

🥥 Modern API for good old MS-COCO datasets.

Features:

- Modern, easy-to-use, neat and tidy API
- Typed codebase with IDE autosuggestions and no "magic" numbers or tuples

## Installation

CocoNuTools is installed like any other PyPi package:

```bash
# install via PIP
pip install coconutools
# or using Poetry
poetry add coconutools
```

## Usage

```python
from pathlib import Path

from coconutools import COCO, Image

dataset = COCO(annotation_file=Path("./tmp/instances_train2014.json"))

print(f"COCO Dataset: {dataset.info.description} ({dataset.info.url})")
print(f"Categories: {','.join([category.name for category in dataset.categories])}")
print(f"Images: {len(dataset.images)}")

for annotation in dataset.annotations:
    image: Image = annotation.image

    print(f"ID #{annotation.id}: {image.width}x{image.height} [{annotation.category.name}]")
```