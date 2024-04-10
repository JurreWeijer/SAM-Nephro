import sys
sys.path.append(".../SAM-Nephro")

import json
from openslide import open_slide 
from pathlib import Path
import numpy as np
import tifffile

from common.conversions import geojson_to_mask
from common.masks import generate_tissue_mask


if __name__ == "__main__":
    
    data_dir = Path("/hpc/dla_patho/home/jweijer/data_clinical_validation/UMC_Utrecht/3M/")

    samples = [folder for folder in data_dir.glob('*') if folder.is_dir()]
    samples.sort()

    for path in samples: 
        
        slide = open_slide(path / "slide.ndpi")
        
        cortex_annotation_path = path / "cortex_annotations.geojson"
        with open(cortex_annotation_path, 'r') as f:
            cortex_data = json.load(f)
            
        dlevel = len(slide.level_dimensions) - 3
        cortex_mask, _ = geojson_to_mask(data=cortex_data, slide=slide, mask_level=dlevel)
        cortex_mask = np.argmax(cortex_mask, axis=2)
        tifffile.imwrite((path / "cortex_tissue_mask.tiff"), cortex_mask.astype(np.uint8), compression= "lzw")
        
        tissue_mask, _ = generate_tissue_mask(slide)
        tifffile.imwrite((path / "tissue_mask.tiff"), tissue_mask.astype(np.uint8), compression= "lzw")