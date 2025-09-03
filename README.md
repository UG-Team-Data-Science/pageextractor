# Page Extractor

Extracts a page from a photo, and warps it to a rectangular image using skimage.

## Features

- Remove margin clutter from photos of pages, usually improved document processing.
- GroundingDINO detection model integration.
- SAM 2.1
- Customizable text prompt, e.g. to "receipt." or "invoice."

## Getting Started

### Prerequisites

- Python 3.10 or higher

### Installation

#### Installing PyTorch Dependencies

Before installing `pageextractor`, install PyTorch:

```bash

pip install torch==2.4.1 torchvision==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu124

```

#### Installation options

```bash

pip install -U git+https://github.com/UG-Team-Data-Science/pageextractor.git

```

Or:

```bash

git clone https://github.com/UG-Team-Data-Science/pageextractor && cd pageextractor
pip install -e .

```

### Usage

```python
from PIL import Image
from matplotlib import pyplot as plt

from pageextractor import PageExtractor

img = Image.open('example.png')
model = PageExtractor(sam_type='sam2.1_hiera_tiny', device='cuda')
mask, polygon, cropped = model.extract_page(img)

_, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(30, 15))
ax0.imshow(img)

ax1.imshow(img)
ax1.plot(*polygon[[0,1,2,3,0]].T, 'r:')
ax1.imshow(1-mask, cmap='Blues', alpha=0.8 - 0.8*mask)

ax2.imshow(cropped)
```

## Acknowledgments

This project is based on/used the following repositories:

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment-Anything](https://github.com/facebookresearch/segment-anything-2)
- [lang-segment-anything](https://github.com/luca-medeiros/lang-segment-anything/)

