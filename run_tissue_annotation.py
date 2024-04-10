import torch
from pathlib import Path
import tifffile
import numpy as np
import json
import argparse

from openslide import open_slide

from tissue_annotation.tissue_annotator import NephroTissueAnnotator
from common.conversions import polygons_to_geojson
from segment_anything import sam_model_registry
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='run inference using the weights obtained from experiments')
    parser.add_argument("data_dir", type=str,)
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--sam_type", type=str, default="vit_h")
    parser.add_argument('--level', type=int, default=0.8, help='magnification level for the extracted patches')
    parser.add_argument('--patch_size', type=int, default=0.85, help='patch width and height for the extracted patches')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.checkpoint_path)
    level = args.level
    patch_size = args.patch_size
    
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device=device)

    # Search for folders with the specified prefixes
    paths = [folder for folder in data_dir.glob("*") if folder.is_dir()]
    paths.sort()
    
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
        annotator.set_slide(slide, None, level, patch_size)
        polygons = annotator.generate_mask()
        
        #convert to geojson
        output_file = path / "annotations.geojson"
        feature_collection = polygons_to_geojson(polygons,level)
        with open(output_file, 'w') as f:
            json.dump(feature_collection, f)
