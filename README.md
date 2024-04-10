# Multiclass Segmentation of Kidney Tissue using the Segment Anything Model
This repository contains the code for an automatic annotation tool based on the Segment Anything Model (SAM) to speed up dataset creation and a modified version of SAM that is fine-tuned on the created dataset. SAM is modfied to facilitate instance segmentation of the following classes: tubulus proximal, tubulus distal, tubulus atrophy, glomerulus, sclerotic glomerulus and vessels. 

![Alt Text](images/model_and_decoder.png)

The figure below shows the annotation tool could be used to speed up dataset creation. The middle image shows the annotations obtained using SAM after which the annotatations can be improved and labeled to obtain a dataset that can be seen in the right image.  

![Alt Text](images/stages_annotation_process.PNG)

The results that can be obtained with the modified version of SAM can be seen in the image below.   

![Alt Text](images/prediction_vs_groundtruth.png)

## Environment

To replicate the exact environment we provide an environment.yaml file containing the specifications of the required packages and their versions. Which can be used as follows to create an environment:

    ```bash
    conda env create -f environment.yml
    ```

## Data annotation using SAM

To create annotations for slides using our annotation tool the tissue_annotation.py file should be used by providing the path to the directory with the slides, the path to the SAM checkpoint and the sam type (e.g. "vit_h", "vit_l" or "vit_b") data an the SAM checkpoint:  

    ```bash
    python tissue_annotation.py "/path/to/your/data_dir" "/path/to/your/checkpoint_path" --sam_type "vit_h" --level 2 --patch_size 1024
    ```

The data directory is expected to contain a folder for each "slide.ndpi" to be annotated, it should work with any slide from any scanner that openslide support (the filename of the slide would have to be changed in the code).  

## Training and Data Preparation

To train the model use the run_train.py file which you provide the path to the configuration file

    ```bash
    python train_sam.py "/path/to/your/config_file"
    ```

To train the model the dataset class expects a data directory that has a folder per sample where each folder contains the slide and the ground truth annotations which should be a geojson file, resulting in the following folder structure:

- data_dir
    - Sample 1
        - slide.ndpi
        - improved_labeled_annotations.geojson
    - Sample 2
        - ...

The dataset should work with slides from the various scanners that openslide supports (slide file name would have to be changed in the code), we do assume that level 2 corresponds to a 10x magnification which we use for training the model. 

## Inference

Inference can be performed with the run_inferece.py file by providing a directory with the slides and the path to the fine-tuned checkpoint. Optionally you can change the IoU threshold, stability threshold, level and patch size but the last two have to be consistent with the values used for training. 

    ```bash
    python run_inference_v2.py "/path/to/your/data_dir" "/path/to/your/checkpoint" --iou_thresh 0.8 --stability_thresh 0.85 --level 2 --patch_size 1024
    ```

The data directory is expected to have one a seperate folder per "slide.ndpi" and optionally a "partial_tissue_mask.tiff" of the cortex. 