import sys
sys.path.append("..")

import torch
from pathlib import Path
import tifffile
import numpy as np
import json

from openslide import open_slide

from slide_segmenter import NephroTissueAnnotator
from common.conversions import polygons_to_geojson
from segment_anything import sam_model_registry
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator



if __name__ == "__main__": 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    DATA_DIR = Path("...") # Add path to data folder where each sample to be labeld has it own folder
    CHECKPOINT_PATH = Path("...") # Ad path to checkpoint
    PATCH_SIZE = 1024
    LEVEL = 2
    
    sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT_PATH)
    sam.to(device=device)

    # Search for folders with the specified prefixes
    paths = [folder for folder in DATA_DIR.glob("*") if folder.is_dir()]
    
    mask_generator = SamAutomaticMaskGenerator(model=sam,
                                               points_per_side=64,
                                               pred_iou_thresh=0.90,
                                               stability_score_thresh=95,
                                               crop_n_layers=1,
                                               min_mask_region_area=100)
    
    annotator = NephroTissueAnnotator(mask_generator=mask_generator)

    for path in paths:
        
        slide = open_slide(path / "slide.ndpi")
        #tissue_mask = tifffile.imread(path/"partial_tissue_mask.tiff")
        
        print("segmenting")
        annotator.set_slide(slide, None, LEVEL, PATCH_SIZE)
        polygons = annotator.generate_mask()
        
        #convert to geojson
        output_file = path / "annotations.geojson"
        feature_collection = polygons_to_geojson(polygons,LEVEL)
        with open(output_file, 'w') as f:
            json.dump(feature_collection, f)
