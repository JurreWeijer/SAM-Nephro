import openslide
import numpy as np
import matplotlib.pyplot as plt
import random
from shapely import Polygon

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image 


from copy import deepcopy
from typing import Tuple

def calc_resolution(patch_size, slide, level, obj, obj_level = None):
    
    # the first object has to be a slide and retrieve dimension 
    if not isinstance(slide,  openslide.OpenSlide):
        raise ValueError("the slide is not openslide object")
    X_slide, _ = slide.level_dimensions[level]
    
    #check wether the second object is slide or array and retrieve dimension
    if isinstance(obj,  openslide.OpenSlide):
        if level is None:
            raise ValueError("level is required for slide objects")
        else: 
            X_obj_2, _ = obj.level_dimensions[obj_level]
    else:
        X_obj_2 = obj.shape[0]

    #calculate the resolutions
    resolution = X_slide / X_obj_2
    patch_size = int(patch_size // resolution)
    
    #ensure that the resolution is a multitude of 2 
    if not np.log2(resolution).is_integer():
            raise Exception(f"Resolution {resolution} is not a power of 2")
    
    return patch_size, resolution

def grid_generator(slide, level, point_spacing):
    grid_points = [(i, j) for i in range(0, int(slide.level_dimensions[level][0]), int(point_spacing)) 
                            for j in range(0, int(slide.level_dimensions[level][1]), int(point_spacing))]
    
    return grid_points

def find_polygons_patch(polygons, patch_polygon, center_patch_polygon=None, shuffle=True):
    if center_patch_polygon is None: 
        center_patch_polygon = patch_polygon
        
    polygons_in_patch = []
    for polygon in polygons:
        if polygon["polygon"].intersects(center_patch_polygon):
            if not polygon["polygon"].is_valid:
                continue
            
            if polygon["label"] == "interstitium":
                continue
            
            intersection = polygon["polygon"].intersection(patch_polygon)
            polygon["polygon"] = intersection
            
            #make sure that there is at least some of the polygon in the patch 
            if intersection.area < 500:
                continue
            
            if isinstance(intersection, Polygon):
                polygons_in_patch.append(polygon)
    
    if shuffle:      
        random.shuffle(polygons_in_patch)
    
    return polygons_in_patch  

def create_patch_polygon(X, Y, slide_resolution, patch_size, overlap=0):
    border = 0.5 * overlap * patch_size
    patch_polygon = Polygon([((X+border)//slide_resolution, (Y+border)//slide_resolution),
                             ((X+patch_size-border)//slide_resolution, (Y+border)//slide_resolution),
                             ((X+patch_size-border)//slide_resolution, (Y+patch_size-border)//slide_resolution),
                             ((X+border)//slide_resolution, (Y+patch_size-border)//slide_resolution)])
    
    return patch_polygon

#####################################################################################################################################
#####################################################################################################################################
# Code below is from SAM

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 
    
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    
class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
