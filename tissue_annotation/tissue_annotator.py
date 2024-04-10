import sys
sys.path.append(".../SAM-Nephro")

import openslide
import cv2
from shapely.geometry import Polygon
import numpy as np

from common.utils import grid_generator
from inference.whole_slide_annotator import from_mask_to_polygon
from common.utils import calc_resolution

class NephroTissueAnnotator():
    def __init__(
        self,
        mask_generator,
    ) -> None:
        super().__init__()
        self.mask_generator = mask_generator
           
    def set_slide(
        self, 
        slide: openslide.OpenSlide, 
        tissue_mask: np.ndarray = None, 
        level: int = 2, 
        patch_size: int = 1048, 
        overlap: float = 0.5,
        ):
        
        self.overlap = overlap
        self.level = level
        self.patch_size = patch_size
        self.point_spacing = self.patch_size*self.overlap
        
        self.slide = slide
        if tissue_mask is None: 
            self.tissue_mask, _ = self.generate_tissue_mask(self.slide)
        else: 
            self.tissue_mask = tissue_mask

        self.tissue_mask_size, self.tissue_resolution = calc_resolution(self.patch_size, self.slide, self.level, self.tissue_mask)
        self.slide_size, self.slide_resolution = calc_resolution(self.patch_size, slide, self.level, slide, 0) 

        self.polygons = []

    def generate_mask(self):
        
        tissue_contours, _ = cv2.findContours(self.tissue_mask.transpose(1,0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        tissue_polygons = []
        for tissue_contour in tissue_contours:
            tissue_contour = np.squeeze(tissue_contour)
            polygon_coords = [(point[0]*self.tissue_resolution, point[1]*self.tissue_resolution) for point in tissue_contour]
            tissue_polygons.append(Polygon(polygon_coords))
        
        for X, Y in grid_generator(self.slide, self.level, self.point_spacing):
            print(f"patch {grid_generator(self.slide, self.level, self.point_spacing).index((X,Y))} out of {len(grid_generator(self.slide, self.level, self.point_spacing))} patches")
            
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
            
            print(f"{len(masks)} number of mask in patch")
            for mask in masks:
                #obtain binary mask and contour
                polygon = from_mask_to_polygon(mask["segmentation"], X, Y)
                if polygon is None:
                    continue
                
                #add the information to the new polygon dictionary
                new_polygon = {}
                new_polygon["polygon"] = polygon 
                new_polygon["predicted_iou"] = mask["predicted_iou"] 
                                     
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
                        intersection = new_polygon["polygon"].intersection(overlapping_polygon["polygon"])
                        overlap_area = intersection.area
                        #first deal with polygons within another polygon
                        if overlap_area > 0.9*overlapping_polygon["polygon"].area or overlap_area > 0.9*new_polygon["polygon"].area:
                            #in this case we pick the largest polgyon
                            if overlapping_polygon["polygon"].area < new_polygon["polygon"].area:
                                #if overlapping_polygon in self.polygons:
                                self.polygons.remove(overlapping_polygon)
                                self.polygons.append(new_polygon)
                        #next check for large overlaps as result of overlapping patches
                        elif overlap_area > 0.1*overlapping_polygon["polygon"].area or overlap_area > 0.1*new_polygon["polygon"].area:
                            self.polygons.append(new_polygon)
                        else:
                            if overlapping_polygon["predicted_iou"] < new_polygon["predicted_iou"]:
                                if overlapping_polygon in self.polygons:
                                    self.polygons.remove(overlapping_polygon)
                                self.polygons.append(new_polygon) 
                    
                # Append the new polygon if no overlap is found
                else:
                    self.polygons.append(new_polygon)  

        return self.polygons
