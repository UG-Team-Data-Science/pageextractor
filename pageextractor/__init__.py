import json
import shapely
import skimage
import rasterio.features
from tqdm.cli import tqdm

import torch
import numpy as np
from PIL import Image
from hydra import compose
from omegaconf import OmegaConf
from hydra.utils import instantiate

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor


def simplify_to_n(polygon, n, eps=1):
  """Simplifies the polygon to have n vertices
  by doing an open-ended binary search of the tolerance needed for
  shapely.simplify. eps is the max extra tolerance needed before
  resulting in a n+1 vertice polygon. E.g. causes more binary search
  iterations but also a tighter fit."""
  assert n >= 3
  
  #remove holes
  polygon = shapely.Polygon(np.array(polygon.exterior.coords.xy).T)
  tolerance = np.sqrt(polygon.area) # not really well thought through start
  # [0, inf) interval, algorithm will reduce this interval through binary
  # search
  min_tolerance = 0
  max_tolerance = None
  simplified = shapely.simplify(polygon, tolerance=tolerance)
  coords = simplified.exterior.coords.xy
  # n+1 because first and last coord are the same, i.e. closed polygon.
  while len(coords[0]) != n+1 or (max_tolerance - min_tolerance) > eps:
    # too coarse, reduce tolerance
    # `<=` is important, when the number of vertices is correct,
    # you want to find the lowest tolerance that wouldn't cause to n+1
    # vertices, e.g. we choose the `==` part to be the branch that
    # lowers the tolerance. Otherwise we'd find the hightest tolerance
    # that would still lead to n vertices instead of `n-1`.
    # Moreover, n-1 isn't always possible, i.e. if someone wants a
    # triangle and sets n=3, `simplify` will never return a line (I think).
    if len(coords[0]) <= n+1:
      max_tolerance = tolerance
      tolerance = (min_tolerance + max_tolerance) / 2
    else:
      # too precise, increase tolerance
      min_tolerance = tolerance
      if max_tolerance is None:
        tolerance *= 2 # just double if no maximum found yet
      else:
        tolerance = (min_tolerance + max_tolerance) / 2
    simplified = shapely.simplify(polygon, tolerance=tolerance)
    coords = simplified.exterior.coords.xy
  return simplified

def aligned_rectangle_coords(polygon):
  """Aligns shapely.Polygon's boundary to clockwise oriantation
  with the the x, y coordinate with x+y minimal first and returns
  as a 4x2 np.array."""
  # Reverse if counter-clockwise
  direction = -1 if polygon.exterior.is_ccw else 1
  # Remove last coordinate as it equals first and ensure direction is clockwise
  coords = np.array(list(polygon.boundary.coords))[:4][::direction]
  # shift first coordinate to be with x+y minimal.
  # this alligns rectangles if you want to average multiple of them.
  # also a canonicality like this  is required if you want to warp
  # to [0,0 -> w,h] coordinates.
  return np.roll(coords, shift=-coords.sum(1).argmin(), axis=0)

SAM_MODELS = {
    "sam2.1_hiera_tiny": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
    },
    "sam2.1_hiera_small": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
    },
    "sam2.1_hiera_base_plus": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    },
    "sam2.1_hiera_large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
    },
}


class PageExtractor():
  def __init__(self, sam_type='sam2.1_hiera_tiny', text_prompt = "page.", gdino_model_id="IDEA-Research/grounding-dino-base", box_threshold = 0.3, text_threshold = 0.25, device=None):
    if device is None:
      device = "cuda" if torch.cuda.is_available() else "cpu"
    self.device = device
    self.text_prompt = text_prompt
    self.box_threshold = box_threshold
    self.text_threshold = text_threshold

    self.processor = AutoProcessor.from_pretrained(gdino_model_id)
    self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_model_id).to(device)

    sam = SAM_MODELS[sam_type]
    cfg = compose(config_name=sam["config"], overrides=[])
    OmegaConf.resolve(cfg)
    self.model = instantiate(cfg.model, _recursive_=True)
    state_dict = torch.hub.load_state_dict_from_url(sam["url"], map_location="cpu")["model"]
    self.model.load_state_dict(state_dict, strict=True)
    self.model = self.model.to(device)
    self.model.eval()
    self.predictor = SAM2ImagePredictor(self.model)


  def extract_page(self, page):
    """Uses SAM2 to extract warp a four-corner around the page area
    and extracts that as the image"""
    inputs = self.processor(images=[img], text=[prompt], padding=True, return_tensors="pt").to(gdino_model.device)

    with torch.no_grad():
        outputs = gdino_model(**inputs)

    results = self.processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, self.box_threshold, text_threshold=self.text_threshold,
        target_sizes=[page.size[::-1]],
    )

    self.predictor.set_image(np.array(img))
    masks, scores, logits = self.predictor.predict(
        box=results[0]['boxes'], multimask_output=False)
    if len(masks.shape) > 3:
      masks = np.squeeze(masks, axis=1)
    mask = masks[0]

    maskShape = list(rasterio.features.shapes(mask.astype('uint8')))
    polygon = max( # extract largest area with mask==1, assume this is the page
      (
        shapely.from_geojson(json.dumps(shape))
        for shape, v in maskShape if v == 1
      ), key=lambda x: x.area
    )
    polygon = simplify_to_n(polygon, 4)
    fourcorner = aligned_rectangle_coords(polygon)
    h, w = map(int, np.sqrt(((fourcorner - fourcorner[[1,2,3,0]]) ** 2).sum(1)).reshape(2,2).mean(0).round())
    src = np.array([[0, 0], [0, h], [w, h], [w, 0]])
    
    tform3 = skimage.transform.ProjectiveTransform()
    tform3.estimate(src, fourcorner)
    cropped = skimage.transform.warp(np.array(img), tform3, output_shape=(h, w))
    cropped = (255 * cropped).astype(np.uint8)
    cropped = Image.fromarray(self.extract_page(page)[2])
    return mask, fourcorner, cropped

  def extract_pages(self, pages):
    """Extracts pages from photo's of pages using SAM.
    Takes a list of PIL.Image and returns PIL.Image's"""
    return [
        {'mask': mask, 'cropped': cropped, 'polygon': fourcorner}
        for page in tqdm(pages)
        for mask, fourcorner, cropped in [self.extract_page(page)]
    ]
