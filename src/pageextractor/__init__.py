import json
import shapely.ops
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

def convexify(polygon):
  """Removes points from the polygon that are far from the convex hull, i.e.
  sqrt(area)/100 as a max dist, again pulled out of shy bass"""
  conv_hull = shapely.convex_hull(polygon)
  dmax = np.sqrt(polygon.area) / 100
  simplified = shapely.make_valid(shapely.Polygon([
      [x,y] for x, y in zip(*polygon.exterior.xy)
      if shapely.Point(x, y).distance(conv_hull.exterior) < dmax
  ]))
  if hasattr(simplified, 'geoms'):
    return max(simplified.geoms, key=lambda x: x.area)
  return simplified

def simplify_to_n(polygon, n, eps=1):
  """Simplifies the polygon to have n vertices
  by doing an open-ended binary search of the tolerance needed for
  shapely.simplify. eps is the max extra tolerance needed before
  resulting in a n+1 vertice polygon. E.g. causes more binary search
  iterations but also a tighter fit."""
  assert n >= 3

  convexified = False

  def f(polygon, n, eps):
    nonlocal convexified
    #remove holes
    polygon = shapely.Polygon(np.array(polygon.exterior.coords.xy).T)
    tolerance = np.sqrt(polygon.area) # not really well thought through start
    # [0, inf) interval, algorithm will reduce this interval through binary
    # search
    min_tolerance = 0
    max_tolerance = None

    # sometimes the algorithm can explode into endlessly increasing the tolerance,
    # yet the polygon won't reduce in points because of weildy, non-convex
    # convavities. E.g. something like this:
    # |‾‾‾‾‾‾‾|
    # |       |
    # |_||    |
    # |‾‾     |
    # |       |
    # |_______|
    # this caps the tolerance to the square root of the area, e.g. a
    # rough estimate of a number that I pulled out of shy bass for the something
    # relative to the circumference of a shape with that area.
    max_min_tolerance = np.sqrt(polygon.area)

    simplified = shapely.simplify(polygon, tolerance=tolerance)
    coords = simplified.exterior.coords.xy
    # n+1 because first and last coord are the same, i.e. closed polygon.
    while len(coords[0]) != n+1 or max_tolerance is None or (max_tolerance - min_tolerance) > eps:
      if min_tolerance > max_min_tolerance:
        if convexified:
          raise ValueError(f"Failed to simplify at tolerance [{min_tolerance}, inf), probably weirdly non-convex")
        convexified = True
        return simplify_to_n(convexify(polygon), n=n, eps=eps)
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
  return f(polygon, n, eps)


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
    self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_model_id).to(self.device)

    sam = SAM_MODELS[sam_type]
    cfg = compose(config_name=sam["config"], overrides=[])
    OmegaConf.resolve(cfg)
    self.model = instantiate(cfg.model, _recursive_=True)
    state_dict = torch.hub.load_state_dict_from_url(sam["url"], map_location="cpu")["model"]
    self.model.load_state_dict(state_dict, strict=True)
    self.model = self.model.to(self.device)
    self.model.eval()
    self.predictor = SAM2ImagePredictor(self.model)


  def extract_page(self, page, prompt=None):
    """Uses SAM2 to extract warp a four-corner around the page area
    and extracts that as the image"""

    prompt = prompt if prompt is not None else self.text_prompt
    inputs = self.processor(images=[page], text=[prompt], padding=True, return_tensors="pt").to(self.device)

    with torch.no_grad():
        outputs = self.gdino_model(**inputs)

    results = self.processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, self.box_threshold, text_threshold=self.text_threshold,
        target_sizes=[page.size[::-1]],
    )

    self.predictor.set_image(np.array(page))
    masks = self.predictor.predict(
        box=results[0]['boxes'], multimask_output=False)[0] #[0] for masks
    if len(masks.shape) > 3:
      masks = np.squeeze(masks, axis=1)
    mask = masks[0] 

    background_white = np.maximum((255-255*mask[..., None]).astype(np.uint8), np.array(page))

    maskShape = list(rasterio.features.shapes(mask.astype('uint8')))
    polygon = max( # extract largest area with mask==1, assume this is the page
      (
        shapely.from_geojson(json.dumps(shape))
        for shape, v in maskShape if v == 1
      ), key=lambda x: x.area
    )
    # remove holes
    polygon = shapely.Polygon(np.stack(polygon.exterior.xy).astype(int).T)
    # fourcorner to map the polygon from or snap the polygon to, e.g. angles
    # might be skewed
    fourcorner = simplify_to_n(polygon, 4)
    if shapely.is_ccw(fourcorner):
      fourcorner = shapely.reverse(fourcorner)
    fourcorner_pts = aligned_rectangle_coords(fourcorner)

    # box to map stuff to, this is actually a box with 90deg angles.
    # h, w as the average of the 4-corner edge lengths
    h, w = map(int, np.sqrt(((fourcorner_pts - fourcorner_pts[[1,2,3,0]]) ** 2).sum(1)).reshape(2,2).mean(0).round())
    src = np.array([[0, 0], [0, h], [w, h], [w, 0]])

    # simplified polygon, otherwise the stretching becomes hella expensive
    simplified = shapely.simplify(polygon, tolerance=np.sqrt(polygon.area)/300)

    # snap the polygon to the 4 corners, e.g. a Nx2x2 array of
    # N polygon points, with the original and snapped point. E.g. `mapping[:,0]`
    # are all points on the polygon and `mapping[:, 1]` are the related points
    # on the fourcorner that are closest
    mapping = np.array([
        [[p.x, p.y] for p in shapely.ops.nearest_points(shapely.Point(x, y), fourcorner.exterior)]
        for x, y in zip(*simplified.exterior.xy)
    ]).astype(int)

    tform0 = skimage.transform.PiecewiseAffineTransform()
    tform0.estimate(mapping[:, 1], mapping[:, 0])

    tform1 = skimage.transform.ProjectiveTransform()
    tform1.estimate(src, fourcorner_pts)

    def chain_tform(t0, t1):
      trans = lambda pts: t0(t1(pts))
      trans.inverse = lambda pts: t1.inverse(t0.inverse(pts))
      return trans

    # only crop the fourcorners and project to a rectangle.
    cropped = skimage.transform.warp(np.array(page), tform1, output_shape=(h, w))
    cropped = (255 * cropped).astype(np.uint8)
    cropped = Image.fromarray(cropped)

    # Alternative 'crop': piecewice affine transform to fourcorners, then crop and
    # project to a rectangle as before. This maps or stretches curved pages inside the crop.
    page_corrections = skimage.transform.warp(np.array(page), chain_tform(tform0, tform1), output_shape=(h, w))

    return mask, fourcorner, polygon, background_white, cropped, page_corrections

  def extract_pages(self, pages, prompt=None):
    """Extracts pages from photo's of pages using SAM.
    Takes a list of PIL.Image and returns PIL.Image's"""

    prompt = prompt if prompt is not None else self.text_prompt
    return [
        {'mask': mask, 'cropped': cropped, 'fourcorner': fourcorner, 'polygon': polygon,
         'white_background': background_white, 'page_corrections': page_corrections}
        for page in tqdm(pages)
        for mask, fourcorner, polygon, background_white, cropped, page_corrections in [
          self.extract_page(page, prompt)
        ]
    ]

  def unload(self):
    """Unload all models from GPU VRAM to free memory."""
    if hasattr(self, 'gdino_model') and self.gdino_model is not None:
      self.gdino_model.to('cpu')
      del self.gdino_model
      self.gdino_model = None
    if hasattr(self, 'model') and self.model is not None:
      self.model.to('cpu')
      del self.model
      self.model = None
    if hasattr(self, 'predictor') and self.predictor is not None:
      del self.predictor
      self.predictor = None
      
    # Clear GPU cache if available
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
