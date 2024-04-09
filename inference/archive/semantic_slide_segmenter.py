import sys
sys.path.append("/hpc/dla_patho/home/jweijer/Internship-Jurre/common")

import openslide
import cv2
from shapely.geometry import Polygon

from skimage.measure import label
import skimage

from models.build_ssam_cnn import SSamCNN
from models.build_sam_class import SamClass
from segment_anything.utils.transforms import ResizeLongestSide
from inference.archive.semantic_automatic_mask_generator import SemanticSamAutomaticMaskGenerator
import numpy as np

from common.utils import calc_resolution

class SemanticSlideSegmenter():
    def __init__(
        self,
        sam_model: SSamCNN,
        classes: list,
        slide: openslide.OpenSlide,
        tissue_mask: np.ndarray = None,
        level: int = 2,
        patch_size: int = 1048,
        overlap: float = 0.5,
    ) -> None:
        super().__init__()
        self.model = sam_model
        self.classes = classes
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.overlap = overlap
        self.level = level
        self.patch_size = patch_size
        self.point_spacing = self.patch_size*self.overlap

        
        self.mask_generator = SemanticSamAutomaticMaskGenerator(
            model=self.model,
            classes=self.classes,
            points_per_side=64,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.80,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )
        
        self.slide = slide
        if tissue_mask is None: 
            self.tissue_mask, _ = self.generate_tissue_mask(self.slide)
        else: 
            self.tissue_mask = tissue_mask

        self.tissue_mask_size, self.tissue_resolution = calc_resolution(self.patch_size, self.slide, self.level, self.tissue_mask)
        self.slide_size, self.slide_resolution = calc_resolution(self.patch_size, slide, self.level, slide, 0) 
        
        self.grid_points = [(i, j) for i in range(0, int(slide.level_dimensions[self.level][0]), int(self.point_spacing)) 
                        for j in range(0, int(slide.level_dimensions[self.level][1]), int(self.point_spacing))]

        self.polygons = []

    def generate_mask(self):
        
        tissue_contours, _ = cv2.findContours(self.tissue_mask.transpose(1,0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        tissue_polygons = []
        for tissue_contour in tissue_contours:
            tissue_contour = np.squeeze(tissue_contour)
            polygon_coords = [(point[0]*self.tissue_resolution, point[1]*self.tissue_resolution) for point in tissue_contour]
            tissue_polygons.append(Polygon(polygon_coords))
        
        for X, Y in self.grid_points:
            tissue_mask_patch = self.tissue_mask[int(X//self.tissue_resolution):int((X//self.tissue_resolution)+self.tissue_mask_size), 
                                            int(Y//self.tissue_resolution):int((Y//self.tissue_resolution)+self.tissue_mask_size)]
            if np.sum(tissue_mask_patch) < 0.05*(self.tissue_mask_size**2):
                continue
            
            image_patch = np.array(self.slide.read_region((int(X/self.slide_resolution),int(Y/self.slide_resolution)), self.level, (self.patch_size,self.patch_size)).convert("RGB"))
            masks = self.mask_generator.generate(image_patch)
            
            center_patch_coords = [(X + 0.5*self.overlap*self.patch_size, Y + 0.5*self.overlap*self.patch_size),
                                   (X + 1.5*self.overlap*self.patch_size, Y + 0.5*self.overlap*self.patch_size),
                                   (X + 1.5*self.overlap*self.patch_size, Y + 1.5*self.overlap*self.patch_size),
                                   (X + 0.5*self.overlap*self.patch_size, Y + 1.5*self.overlap*self.patch_size)]

            center_patch = Polygon(center_patch_coords)
            
            for mask in masks:
                #obtain binary mask and contour
                binary_mask = mask["segmentation"].astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(contours) < 1: 
                    continue
                contour = contours[0]
                contour = np.squeeze(contour)
                if len(contour) < 10: 
                    continue
                
                #retrieve the coordinates of the polygon
                polygon_coords = [(point[0] + X, point[1] + Y) for point in contour]
                    
                #add the information to the new polygon dictionary
                new_polygon = {}
                new_polygon["polygon"] = Polygon(polygon_coords)
                new_polygon["predicted_iou"] = mask["predicted_iou"] 
                
                if mask["label"] != -1: 
                    new_polygon["label"] = self.classes[mask["label"]] 
                else:
                    new_polygon["label"] = "background"
                                     
                
                #tresholds for size of the mask
                if not new_polygon["polygon"].is_valid:
                    continue
                elif new_polygon["polygon"].area < 100 or new_polygon["polygon"].area > 0.04*(self.patch_size**2):
                    continue
                elif not new_polygon["polygon"].intersects(center_patch):
                    continue
                elif not any(new_polygon["polygon"].within(tissue_polygon) for tissue_polygon in tissue_polygons):
                    continue
                
                #find all the overlapping polygons
                overlapping_polygons = []
                for existing_polygon in self.polygons:
                    if new_polygon["polygon"].intersects(existing_polygon["polygon"]):
                        overlapping_polygons.append(existing_polygon)
                    
                #deal with overlap
                if len(overlapping_polygons) > 0:
                    for overlapping_polygon in overlapping_polygons:
                        #overlapping_polygon = overlapping_polygons[0]
                        intersection = new_polygon["polygon"].intersection(overlapping_polygon["polygon"])
                        overlap_area = intersection.area
                        #first deal with polygons within another polygon
                        if overlap_area > 0.9*overlapping_polygon["polygon"].area or overlap_area > 0.9*new_polygon["polygon"].area:
                            #in this case we pick the largest polgyon
                            if overlapping_polygon["polygon"].area < new_polygon["polygon"].area:
                                if overlapping_polygon in self.polygons:
                                    self.polygons.remove(overlapping_polygon)
                                self.polygons.append(new_polygon)
                        #next check for large overlaps as result of overlapping patches
                        elif  overlap_area > 0.5*overlapping_polygon["polygon"].area or overlap_area > 0.5*new_polygon["polygon"].area:
                            #in this case we pick the polygon with the highest predicted iou
                            if overlapping_polygon["predicted_iou"] < new_polygon["predicted_iou"]:
                                if overlapping_polygon in self.polygons:
                                    self.polygons.remove(overlapping_polygon)
                                self.polygons.append(new_polygon)
                        #deal with the smaller overlaps
                        else: 
                            #if the new polygons overlaps with multiple polygons we do not add it to the list of polygons
                            if len(overlapping_polygons) > 1:
                                continue 
                            #we pick the largest of the two masks
                            if overlapping_polygon["polygon"].area < new_polygon["polygon"].area:
                                if overlapping_polygon in self.polygons:
                                    self.polygons.remove(overlapping_polygon)
                                self.polygons.append(new_polygon)
                            else:
                                pass
                    
                # Append the new polygon if no overlap is found
                else:
                    self.polygons.append(new_polygon)  

        return self.polygons
    
    def generate_tissue_mask(self, slide):
        """Creates a mask out of a WSI by applying a Gaussian filter to blur the image and consequently apply Otsu's thresholding to it

        Args:
            slide (OpenSlide): an OpenSlide object

        Returns:
            mask: a 2D map of the mask
            pixels: non-zero coordinates of the mask
            downsample
        """
        # Thresholding
        dlevel = len(slide.level_dimensions) - 3
        downsample = slide.level_downsamples[dlevel]
        overview = slide.read_region((0, 0), dlevel, slide.level_dimensions[dlevel])
        img = np.array(overview)
        black_pixels = np.where(
            (img[:, :, 0] <50) & 
            (img[:, :, 1] <50) & 
            (img[:, :, 2] <50)
        )

        # set those pixels to white
        img[black_pixels] = [240, 240, 240, 0]
        img = img[:, :, ::-1].copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.bitwise_not(th3)

        # Remove small islands
        labelMap = label(mask)
        icc = len(np.unique(labelMap))
        for i in range(icc):
            if (
                np.sum(labelMap == i) < 500
            ):  # cluster with a number of pixels smaller than this threshold will be set to zero
                mask[labelMap == i] = 0

        # Find pixels
        disk = skimage.morphology.disk(7) 
        mask = skimage.morphology.binary_dilation(mask, footprint=disk)
        mask = mask.transpose(1,0) 

        pixels = np.nonzero(mask)
        return mask, pixels