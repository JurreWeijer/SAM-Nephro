import sys
sys.path.append("/hpc/dla_patho/home/jweijer/Internship-Jurre")

import torch
from pathlib import Path
import tifffile
import numpy as np
import json

from openslide import open_slide

from inference.labeled_slide_annotator import NeprhoTissueLabeledAnnotator
from inference.automatic_labeled_mask_generator import SamAutomaticLabeledMaskGenerator
from inference.predictor_classification import ClassificationSamPredictor
from common.conversions import labeled_polygons_to_geojson 
from models.build_sam_class import build_samclass_vit_b
import argparse

CLASSES = ["tubulus proximal",
          "tubulus distal",
          "tubulus atrophy",
          "glomerulus",
          "glomerulosclerosis",
          "vessel",
          "background",
          ]

def load_model(checkpoint):
    
    with open(checkpoint, "rb") as f:
        state_dict = torch.load(f, map_location=torch.device('cpu'))

    model = build_samclass_vit_b(classes=CLASSES) 
        
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

def create_automatic_generator(model, iou_tresh, stability_tresh):
    
    generator = SamAutomaticLabeledMaskGenerator(
            predictor=ClassificationSamPredictor(model),
            classes=CLASSES,
            points_per_side=64,
            pred_iou_thresh=iou_tresh,
            stability_score_thresh=stability_tresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )
    
    return generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run inference using the weights obtained from experiments')
    parser.add_argument("data_dir", type=str,)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument('--iou_thresh', type=int, default=0.8, help='treshold for the predicted iou during automatic mask generation')
    parser.add_argument('--stability_thresh', type=int, default=0.85, help='treshold for the stability score during automatic mask generation')
    parser.add_argument('--level', type=int, default=2, help='magnification level of the extracted patches, keep consitent with training values')
    parser.add_argument('--patch_size', type=int, default=1024, help='patch size of the extracted patches, keep consistent with training values')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    level = args.level
    patch_size = args.patch_size

    print("load model")
    model = load_model(args.checkpoint)
    model.to(device)  
    
    print("create mask generator")
    mask_generator = create_automatic_generator(model, args.iou_thresh, args.stability_thresh)
    slide_annotator = NeprhoTissueLabeledAnnotator(mask_generator=mask_generator, classes=CLASSES)
    
    samples = [folder for folder in args.data_dir.glob('*') if folder.is_dir()]
    samples.sort()
    
    print(f"inference performed on {len(samples)} slides")
    
    for path in samples:
        print(f"current path: {path}")
        
        print("open slide and tissue mask")
        slide = open_slide(path / "slide.ndpi")
        
        try: 
            tissue_mask = tifffile.imread(path / "partial_tissue_mask.tiff")
        except: 
            tissue_mask = None   
        
        print("generate annotations")
        slide_annotator.set_slide(slide=slide, tissue_mask=tissue_mask, level=level, patch_size=level)
        polygons = slide_annotator.generate_mask()
        
        
        feature_collection = labeled_polygons_to_geojson(polygons, level)
            
        output_file = path / f"automatic_inferred_annotations.geojson"
        with open(output_file, 'w') as f:
            json.dump(feature_collection, f)
