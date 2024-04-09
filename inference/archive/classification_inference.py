import sys
sys.path.append("/hpc/dla_patho/home/jweijer/Internship-Jurre")

import torch
from pathlib import Path
import tifffile
import numpy as np
import json

from openslide import open_slide

from inference.whole_slide_annotator import WholeSlideAnnotator
from automatic_labeled_mask_generator import SamAutomaticLabeledMaskGenerator
from inference.predictor_classification import ClassificationSamPredictor
from common.conversions import labeled_polygons_to_geojson 
from models.build_sam_class import build_samclass_vit_b, build_samclass_vit_h, build_samclass_vit_l
from fine_tuning_classification.extra_encoders import ResnetSamEncoder

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    experiment = "127_including_other_freeze_sam_tuned_weights_ssam_25_epochs"
    experiment_num = experiment.split("_")[0]

    base_dir = Path("/hpc/dla_patho/home/jweijer/Internship-Jurre/results_binary/")
    experiment_dir = base_dir / experiment

    with open(experiment/ "hyperparameters.json", 'r') as f:
        hyperparameters = json.load(f)

    classes = hyperparameters["classes"]
    #CLASSES[0] = "other"
    level = hyperparameters["level"]
    patch_size = hyperparameters["patch size"]

    additional_encoder = hyperparameters["extra_encoder"]
    sam_type = hyperparameters["sam type"]

    if additional_encoder is None:
        cnn_encoder = None
    elif additional_encoder == "ResnetSamEncoder":
        cnn_encoder = ResnetSamEncoder()
    else:
        raise ValueError(f"Unsupported cnn encoder: {additional_encoder}")

    #sam_checkpoint = "/hpc/dla_patho/home/jweijer/checkpoints/full_model_cnn_80.pth"

    checkpoint = experiment_dir / "sam_model_best.pth" 

    if sam_type == "vit-h":
        sam_model = build_samclass_vit_h(cnn_encoder=cnn_encoder,
                                        classes=classes,
                                        freeze_image_encoder=True,
                                        freeze_cnn_encoder=True,
                                        freeze_prompt_encoder=True,
                                        freeze_mask_decoder=True,
                                        freeze_class_decoder=True,
                                        checkpoint=None,
                                        )
    elif sam_type == "vit-l":
        sam_model = build_samclass_vit_l(cnn_encoder=cnn_encoder,
                                        classes=classes,
                                        freeze_image_encoder=True,
                                        freeze_cnn_encoder=True,
                                        freeze_prompt_encoder=True,
                                        freeze_mask_decoder=True,
                                        freeze_class_decoder=True,
                                        checkpoint=None,
                                        )
    elif sam_type == "vit-b":
        sam_model = build_samclass_vit_b(cnn_encoder=cnn_encoder,
                                        classes=sam_type,
                                        freeze_image_encoder=True,
                                        freeze_cnn_encoder=True,
                                        freeze_prompt_encoder=True,
                                        freeze_mask_decoder=True,
                                        freeze_class_decoder=True,
                                        checkpoint=None,
                                        )
        
    with open(checkpoint, "rb") as f:
        state_dict = torch.load(f, map_location=torch.device('cpu'))

    sam_model.load_state_dict(state_dict)
    sam_model.eval()
    sam_model.to(device)

    data_dir = Path("/hpc/dla_patho/home/jweijer/data_kidney_PAS")
    #data_dir = Path("/hpc/dla_patho/home/jweijer/chicken_embryos/")
    # List of prefixes to search for
    prefixes_to_search = ["0"]
    paths = []

    # Search for folders with the specified prefixes
    for prefix in prefixes_to_search:
        folder_pattern = f"{prefix}*"
        matching_folders = data_dir.glob(folder_pattern)
        paths.extend(filter(lambda p: p.is_dir(), matching_folders))

    #paths = [path for path in DATA_DIR.glob("*")]
    paths.sort()
    print(paths)
    
    mask_generator = SamAutomaticLabeledMaskGenerator(
            predictor=ClassificationSamPredictor(sam_model),
            classes=classes,
            points_per_side=64,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.80,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )

    for path in paths:

        #ndpi_files = list(path.glob('*.ndpi'))
        #ndpi_file_path = ndpi_files[0]
        slide = open_slide(path / "slide.ndpi")
        tissue_mask = tifffile.imread(path/"kidney_mask.tiff")
        
        print("creating segmenter")
        segmenter = WholeSlideAnnotator(mask_generator=mask_generator, classes=classes, slide=slide, tissue_mask=tissue_mask, level=level, patch_size=patch_size)
        print("segmenting")
        polygons = segmenter.generate_mask()
        
        #convert to xml
        #tree = polygons_to_xml(polygons, LEVEL)
        #tree.write(path / "annotations.xml", xml_declaration=True)
        
        #convert to geojson
        output_file = path / f"labeled_annotations_{experiment_num}.geojson"
        feature_collection = labeled_polygons_to_geojson(polygons,level)
        with open(output_file, 'w') as f:
            json.dump(feature_collection, f)
            
        #convert to geojson
        #output_file = path / "tuned_annotations.geojson"
        #feature_collection = polygons_to_geojson(polygons,LEVEL)
        #with open(output_file, 'w') as f:
        #    json.dump(feature_collection, f)
