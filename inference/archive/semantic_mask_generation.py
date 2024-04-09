import sys
sys.path.append("/hpc/dla_patho/home/jweijer/Internship-Jurre")

import torch
from pathlib import Path
import tifffile
import numpy as np
import json

from openslide import open_slide

from inference.archive.semantic_slide_segmenter import SemanticSlideSegmenter
from common.conversions import polygons_to_xml, labeled_polygons_to_geojson, polygons_to_geojson
from models.build_ssam_cnn import build_ssamcnn_vit_b
from fine_tuning_classification.archive.extra_decoders import HovernetNCSmall
from fine_tuning_classification.extra_encoders import ResnetSamEncoder
import sys
sys.path.append("..")
from segment_anything import sam_model_registry

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

classes = ["other",
           "tubulus proximal",
           "tubulus distal",
           "tubulus atrophy",
           "glomerulus",
           "glomerulosclerosis",
           #"peritubular capillary",
           "vessel",
]

semantic_checkpoint = "/hpc/dla_patho/home/jweijer/checkpoints/semantic_model_75.pth" 
sam_checkpoint = "/hpc/dla_patho/home/jweijer/checkpoints/sam_vit_b_01ec64.pth"

sam_model = build_ssamcnn_vit_b(cnn_encoder=ResnetSamEncoder(),
                                semantic_decoder=HovernetNCSmall(5, len(classes)),
                                freeze_image_encoder=True,
                                freeze_cnn_encoder=True,
                                freeze_prompt_encoder=True,
                                freeze_mask_decoder=True,
                                freeze_semantic_decoder=True,
                                sam_checkpoint=sam_checkpoint,
                                semantic_checkpoint=semantic_checkpoint)

sam_model.eval()
sam_model.to(device)

PATCH_SIZE = 1024
LEVEL = 2

data_dir = Path("/hpc/dla_patho/home/jweijer/data_kidney_PAS/Biopsies_3m")
# List of prefixes to search for
prefixes_to_search = ["T"]
paths = []

# Search for folders with the specified prefixes
for prefix in prefixes_to_search:
    folder_pattern = f"{prefix}*"
    matching_folders = data_dir.glob(folder_pattern)
    paths.extend(filter(lambda p: p.is_dir(), matching_folders))

#paths = [path for path in DATA_DIR.glob("*")]
paths.sort()
print(paths)

for path in paths:

    ndpi_files = list(path.glob('*.ndpi'))
    ndpi_file_path = ndpi_files[0]
    slide = open_slide(ndpi_file_path)
    #tissue_mask = tifffile.imread(path/"partial_tissue_mask.tiff")
    
    print("creating segmenter")
    segmenter = SemanticSlideSegmenter(sam_model=sam_model, classes=classes, slide=slide, tissue_mask=None, level=LEVEL, patch_size=PATCH_SIZE)
    print("segmenting")
    polygons = segmenter.generate_mask()
    
    #convert to xml
    #tree = polygons_to_xml(polygons, LEVEL)
    #tree.write(path / "annotations.xml", xml_declaration=True)
    
    #convert to geojson
    output_file = path / "labeled_annotations.geojson"
    feature_collection = labeled_polygons_to_geojson(polygons,LEVEL)
    with open(output_file, 'w') as f:
        json.dump(feature_collection, f)
        
    #convert to geojson
    #output_file = path / "tuned_annotations.geojson"
    #feature_collection = polygons_to_geojson(polygons,LEVEL)
    #with open(output_file, 'w') as f:
    #    json.dump(feature_collection, f)
