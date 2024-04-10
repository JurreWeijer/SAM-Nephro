import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
import random
import albumentations as A
import yaml
import shutil
import argparse

import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

from fine_tuning.trainer import ClassificationTrainer
from fine_tuning.dataset import KidneyClassificationWSIDataset
from fine_tuning.loss import  DiceBCELoss

#from segment_anything.build_sam import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
from models.build_sam_class import build_samclass_vit_b, build_samclass_vit_l, build_samclass_vit_h
#set device 

model_registry =  {'vit-h': build_samclass_vit_h,
                   'vit-l': build_samclass_vit_l,
                   'vit-b': build_samclass_vit_b,
                }

sam_checkpoints = {'vit-h': ".../sam_vit_h_4b8939.pth",
                   'vit-l': ".../sam_vit_l_0b3195.pth",
                   'vit-b': ".../sam_vit_b_01ec64.pth",
                }

optimizer_registry = {"Adam": Adam,
                      "AdamW": AdamW}

loss_registry = {"DiceBCELoss": DiceBCELoss,
                 "CE": torch.nn.CrossEntropyLoss,
                 "MSE": torch.nn.MSELoss,
                }


def load_model(config, parameters_to_exclude):
    
    if config['model_checkpoint'] == None: 
        model_checkpoint = sam_checkpoints[config['sam_type']]
    else:
        model_checkpoint = None
    
    model = model_registry[config['sam_type']](classes=config['classes'],
                                               freeze_image_encoder=config['freeze_image_encoder'],
                                               freeze_prompt_encoder=config['freeze_prompt_encoder'],
                                               freeze_mask_decoder=config['freeze_mask_decoder'],
                                               checkpoint=model_checkpoint,
                                            ) 
    
    
    if config['model_checkpoint'] is not None:
        checkpoint = config['model_checkpoint']
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device('cpu'))
        
        # Filter out the parameters
        filtered_state_dict = {name: param for name, param in state_dict.items() if name not in parameters_to_exclude}
        
        model.load_state_dict(filtered_state_dict, strict=False)
    
    return model


def load_data(config, paths, sampling):
    
    dataset = KidneyClassificationWSIDataset(paths=paths,
                                             level=config['level'],
                                             patch_size=config['patch_size'], 
                                             sampling_classes=sampling,
                                             num_progress_images=config["num_progress_images"],
                                             seed=config['seed'],
                                             include_original=config['include_original'],
                                             create_new=config['create_new'],
                                            )
    
    dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=config['shuffle_dataloader'],
            pin_memory=config['pin_memory_dataloader'],
            num_workers=2,
        )
    
    return dataset, dataloader

def load_trainer(model, train_loader, valid_loader, progress_dir):
    
    warmup_period = config['warmup_epochs'] * (len(train_loader.dataset) / config['batch_size'])
    if warmup_period > 0: 
        base_lr = config['learning_rate']/warmup_period
    else: 
        base_lr = config['learning_rate']
        
    optimizer = optimizer_registry[config['optimizer']](model.parameters(), lr=base_lr, weight_decay=config['weight_decay'])
    segmentation_loss = loss_registry[config['segmentation_loss']](config['dice_weight'])
    classification_loss = loss_registry[config['classification_loss']]()
    iou_loss = loss_registry[config['iou_loss']]()
    
    trainer = ClassificationTrainer(model=model,
                                    segmentation_loss=segmentation_loss,
                                    classification_loss=classification_loss,
                                    iou_loss=iou_loss,
                                    segmentation_weight=config['segmentation_weight'],
                                    classification_weight=config['classification_weight'],
                                    iou_weight=config['iou_weight'],
                                    optimizer=optimizer,
                                    train_loader=train_loader,
                                    valid_loader=valid_loader,
                                    classes=config['classes'],
                                    accumulation_steps=config['accumulation_steps'],
                                    progress_dir=progress_dir,
                                    seed=config['seed'],
                                    warmup_period=warmup_period,
                                    lr_decay=None,
                                    tolerance=config['tolerance'], 
                                    progress_images=config['progress_images'],
                                    display_freq=config['display_freq'],
                                    early_stopping=config['early_stopping'],
                        )
    
    return trainer

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config

def save_config(result_dir, config):
    os.path.join(result_dir, config_name)
    with open(result_dir/'config.yaml', 'w') as file:
        yaml.dump(config, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train modified version of SAM for instance segmentation')
    parser.add_argument("config_path", type=str, help="path to the config file")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config_path = args.config_path
    
    print(f"load config file")
    config_name = "classification_configuration_new.yaml"
    config = load_config(config_path)

    #set manual seed
    torch.manual_seed(config['seed']) 
    torch.cuda.manual_seed(config['seed'])
    random.seed(config['seed'])
    
    if config['cnn_encoder'] is not None: 
        assert config['freeze_cnn_encoder'] == False, "cnn encoder used but frozen"
    
    if config['freeze_mask_decoder'] == 'segmentation':
        assert config["segmentation_weight"] == 0.0, "set to training segmentation but loss weight are incorrect"
         
    if config['freeze_mask_decoder'] == 'classification':
        assert config['classification_weight'] == 0.0, "set to training classification but loss weight are incorrect"
        
    progress_dir = Path(config['results_dir']) / f"{config['task']}_{config['epochs']}_epochs"
    if not os.path.exists(progress_dir):
        os.makedirs(progress_dir)
    
    shutil.copy(config_path, progress_dir/'config.yaml')
    
    print("load model") 
    ### some parameters might have to be removed when starting training from a checkpoint 
    parameters_to_exclude = []

    model = load_model(config, parameters_to_exclude)
    model.to(device)
    
    # Search for folders with the specified prefixes
    data_dir = Path(config['data_dir'])
    sample_dir = data_dir / config['dataset'] / "train+val"
    samples = [folder for folder in sample_dir.glob('*') if folder.is_dir()]
    samples.sort()
    print(f"num samples: {len(samples)}")
    
    train_paths, valid_paths = train_test_split(samples, test_size=config["valid_size"], random_state=config["seed"])
    
    print(f"load dataset")
    train_dataset, train_loader = load_data(config, train_paths, config['train_sampling'])
    valid_dataset, valid_loader = load_data(config, valid_paths, config['valid_sampling'])
    
    total_train_original = 0
    for key in train_dataset.number_original_samples:
        total_train_original += train_dataset.number_original_samples[key]
    
    total_valid_original = 0
    for key in valid_dataset.number_original_samples:
        total_valid_original += valid_dataset.number_original_samples[key]
    
    print("save dataset nubers")
    sample_numbers = {"total number of train samples": len(train_dataset),
                      "total number of valid samples": len(valid_dataset),
                      "total number of original train samples": total_train_original,
                      "total number of original valid samples": total_valid_original,
                      "number of original train samples": train_dataset.number_original_samples,
                      "number of original valid samples": valid_dataset.number_original_samples,
                      "number of train samples": train_dataset.number_total_samples,
                      "number of valid samples": valid_dataset.number_total_samples,
                      "number of skipped train samples": train_dataset.number_skipped_samples,
                      "number of skipped valid samples": valid_dataset.number_skipped_samples,}

    with open(progress_dir / 'sample_number.json', 'w') as j:
        json.dump(sample_numbers, j)

    print("load trainer")
    trainer = load_trainer(model, train_loader, valid_loader, progress_dir)
    
    #train the model
    print(f"train the model")
    trainer.train(max_epochs=config['epochs'])
