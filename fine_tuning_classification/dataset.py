
import sys
sys.path.append("..")

from openslide import open_slide
import numpy as np 
import torch 
from shapely.geometry import Polygon, MultiPolygon, Point
import tifffile
import json
import cv2
import random
import albumentations as A
import shutil

from common.utils import find_polygons_patch, create_patch_polygon

from common.utils import calc_resolution, ResizeLongestSide
from common.conversions import geojson_to_polygon
from trainer import generate_prompts

class KidneyClassificationWSIDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        paths: list,  
        level: int,
        patch_size: int,   
        sampling_classes: dict,
        num_progress_images: int, 
        seed: int,
        include_original: bool = True, 
        create_new: bool = False,
        ):
        
        """
        Custom dataset class for kidney classification in Whole Slide Images (WSI).

        Args:
            paths (list): List of paths to directories containing WSI data and it annotations in a geojson file.
            level (int): Magnification level for patch extraction.
            patch_size (int): Size of the square patches to be extracted.
            sampling_classes (dict): Dictionary specifying the number of augmented samples to be generated per original sample for each class.
            seed (int): Seed for random number generation.
            include_original (bool, optional): Include original samples without augmentation. Default is True.
            create_new (bool, optional): Create a new data folder even if it exists. Default is False.
        """
        
        # Initialize the dataset with necessary parameters
        self.level = level
        self.patch_size = patch_size
        self.sampling_classes = sampling_classes
        self.classes = [key for key in sampling_classes.keys()]
        self.include_original = include_original
        self.create_new = create_new
        self.minimum_mask_size = 100
        self.num_progress_images = num_progress_images
        print(f"self.num_progres_images: {self.num_progress_images}")
        
        self.samples = []
        self.number_original_samples = {cls:0 for cls in self.classes}
        self.number_total_samples = {cls:0 for cls in self.classes}
        self.number_skipped_samples = {cls:0 for cls in self.classes}
        
        self.progress_images = []
        self.progress_masks = []
        self.progress_prompts = []
        
        self.spatial_transform = A.Compose([
            A.Flip(p=1),
            A.Rotate(limit=[-180,180]),
            A.ElasticTransform(p=1, alpha=80, sigma=120 * 0.05, alpha_affine=120 * 0.03),  
        ])
        
        self.color_transform = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.GaussianBlur(p=0.5),
            A.GaussNoise(p=0.5)    
        ])
        
        # Create object for resizing from SAM
        self.resize = ResizeLongestSide(1024) #TODO: this is not very neat
        random.seed(seed)
        
        for path in paths:
            # Open the NDPI slide and read the partial tissue mask
            slide = open_slide(path / "slide.ndpi")
            _, slide_resolution = calc_resolution(self.patch_size, slide, self.level, slide, 0)
            
            # Load labeled annotations from the adapted geojson file
            json_file = path / "improved_labeled_annotations.geojson"
            with open(json_file, 'r') as f:
                mask_data = json.load(f)
            
            # Convert Geojson to Shapely Polygons   
            polygons = geojson_to_polygon(data=mask_data, classes=self.classes, without_holes=True, without_multi=True)
            
            # Create the folder name
            folder_name = f"data_level_{self.level}_classes_{len(self.classes)}"
            folder_path = path / folder_name
            print(f"folder name: {folder_name}")
            
            # Check if folder path exist and whether to create a new folder
            if folder_path.exists() and folder_path.is_dir() and self.create_new is False:
                
                image_files = list(folder_path.glob("*image*"))
                gt_mask_files = list(folder_path.glob("*gt_mask*"))
                gt_label_files = list(folder_path.glob("*gt_label*"))
                
                print(f"image files: {len(image_files)}")
                print(f"mask files: {len(gt_mask_files)}")
                print(f"label files: {len(gt_label_files)}")
                
                assert len(image_files) == len(gt_mask_files) and len(image_files) == len(gt_label_files), "not all images have a mask or label"
                #assert len(gt_label_files) == len(polygons), "polygons are missing in the data folder"
                
                ## Validate that the data in the folder is correct
                #if len(gt_label_files) != len(polygons):
                #    # If the data in the folder is incorrect create new folder
                #    self._create_data(folder_path, polygons, slide, slide_resolution)
                #elif len(image_files) != len(gt_mask_files) and len(image_files) != len(gt_label_files):
                #    # If the data in the folder is incorrect create new folder
                #    self._create_data(folder_path, polygons, slide, slide_resolution)
                    
                # If the data in the folder is correct use it
                print(f"file path exists for folder: {path}")
                self._data_from_folder(folder_path, polygons)
            else:
                # If folder path does not exist or create new then create a new data folder and save the data
                print(f"file path does not exist for folder: {path}")
                self._create_data(folder_path, polygons, slide, slide_resolution)

            print(f"number of progress images: {len(self.progress_images)}")
        
        # Concatenate the progress images, prompts and masks
        self.progress_images = np.concatenate(self.progress_images, axis=0)
        self.progress_masks = np.concatenate(self.progress_masks, axis=0)
        self.progress_prompts = np.concatenate(self.progress_prompts, axis=0)  
        
        print(f"progress_images shape: {self.progress_images.shape}")
        print(f"progress_masks shape: {self.progress_masks.shape}")
        print(f"progress_prompts shape: {self.progress_prompts.shape}")
        
    def _data_from_folder(self, folder_path, polygons):
        """
        Load data from an existing data folder.

        Args:
            folder_path: Path to the data folder.
            polygons: List of Shapely polygons corresponding to the data.

        Returns:
            None
            
        """
        gt_label_files = list(folder_path.glob("*gt_label*"))
        
        for file_path in gt_label_files:
            file_name = file_path.name
            file_number = file_name.split('_')[0]
            
            gt_label = tifffile.imread(file_path)
            label_value = np.argmax(gt_label, axis=1)
            label_name = self.classes[label_value[0]]
            
            if self.include_original: 
                self.samples.append({"idx": file_number,
                                     "folder path" : folder_path,
                                     "augmentation" : False
                                    })
                self.number_original_samples[label_name] += 1
                self.number_total_samples[label_name] += 1
                
                
            for i in range(self.sampling_classes[label_name]):
                self.samples.append({"idx": file_number,
                                     "folder path" : folder_path,
                                     "augmentation" : True
                                    })
                self.number_total_samples[label_name] += 1
        
        progress_image_path = folder_path / "progress_arrays.npz"
        #progress_image_path = folder_path / "progress_i.tiff"
        if progress_image_path.exists() and len(self.progress_images) < self.num_progress_images:
            progress_data = np.load(progress_image_path)
            self.progress_images.append(progress_data['image'])
            self.progress_masks.append(progress_data['mask'])
            self.progress_prompts.append(progress_data['prompts'])
        
    def _create_data(self, folder_path, polygons, slide, slide_resolution):
        """
        Create a new data folder and save data.

        Args:
            folder_path: Path to the new data folder.
            polygons: List of Shapely polygons corresponding to the data.
            slide: WSI slide object.
            slide_resolution: Resolution of the slide.

        Returns:
            None
            
        """
        if folder_path.exists():
            shutil.rmtree(folder_path)
        
        folder_path.mkdir(parents=True)
        
        progress_polygon = False
        for idx, polygon in enumerate(polygons):
            
            # Check if the polygon is a shapely Polygon
            if not isinstance(polygon['polygon'], Polygon):
                self.number_skipped_samples[polygon['label']]
                continue
            
            # Find patch around polygon
            centroid = polygon["polygon"].centroid
            distances = [centroid.distance(Point(x, y)) for x, y in polygon["polygon"].exterior.coords]
            radius = int(max(distances))
            
            if radius < 0.5*self.patch_size:
                X = int(centroid.x*slide_resolution - random.randint(radius, (self.patch_size-radius))) 
                Y = int(centroid.y*slide_resolution - random.randint(radius, (self.patch_size-radius)))
            else: 
                X = int(centroid.x*slide_resolution) - 0.5*self.patch_size
                Y = int(centroid.y*slide_resolution) - 0.5*self.patch_size 
            
            # Create a label tensor which is a one-hot tensor of the class
            label_name = polygon["label"]
            cls_index = self.classes.index(label_name)
            label_np = torch.zeros(len(self.classes))
            label_np[cls_index] = 1 
        
            # Save for progress Image
            if label_name == "glomerulus" and progress_polygon is False: 
                progress_polygon = True
                progress_image, progress_mask, progress_prompt = self._create_progress_images(slide, polygon, centroid, slide_resolution, polygons)
                np.savez( folder_path / 'progress_arrays.npz', image=progress_image, mask=progress_mask, prompts=progress_prompt)
                if len(self.progress_images) < self.num_progress_images:
                    self.progress_images.append(progress_image)
                    self.progress_masks.append(progress_mask)
                    self.progress_prompts.append(progress_prompt)
                print(f"len progress images: {len(self.progress_images)}")
            
            # Read and resize the slide patch
            slide_patch = np.array(slide.read_region((int(X/slide_resolution),int(Y/slide_resolution)), self.level, (self.patch_size,self.patch_size)).convert("RGB"))
            slide_patch_resized = self.resize.apply_image(slide_patch)
            slide_patch_np = slide_patch_resized.transpose(2,0,1)[None, :, :, :]
            
            # Create the mask from the polygon 
            mask_patch = self._draw_mask_from_poly(X, Y, polygon)
            mask_patch_np = mask_patch.transpose(2,0,1)[None, :, :, :]
            if np.count_nonzero(mask_patch_np) < self.minimum_mask_size:
                self.number_skipped_samples[label_name] += 1 
                continue
            
            input = slide_patch_np
            gt_mask = mask_patch_np
            gt_label = np.expand_dims(label_np, axis=0)
                
            if self.include_original:
                self.samples.append({"idx" : idx,
                                        "folder path" : folder_path,
                                        "augmentation" : False
                                    })
                self.number_original_samples[label_name] += 1
                self.number_total_samples[label_name] += 1
                
            for i in range(self.sampling_classes[label_name]):
                self.samples.append({"idx" : idx,
                                        "folder path" : folder_path,
                                        "augmentation" : True
                                    })
                self.number_total_samples[label_name] += 1
                
                
            tifffile.imwrite((folder_path / f"{idx}_image.tiff"), input, compression= "lzw")
            tifffile.imwrite((folder_path / f"{idx}_gt_mask.tiff"), gt_mask.astype(np.uint8), compression= "lzw")
            tifffile.imwrite((folder_path / f"{idx}_gt_label.tiff"), gt_label.astype(np.uint8), compression= "lzw")
            
    
    def _create_progress_images(self, slide, polygon, centroid, slide_resolution, polygons):
        """
        Create progress images for visualization, containing the original slide patch,
        labeled masks of selected polygons, and generated prompts.

        Args:
            slide (WSI slide object): Whole Slide Image object.
            polygon (Shapely polygon): Shapely Polygon chosen to be the polygon in the center of the patch.
            centroid (Point): Centroid of the original selected polygon.
            slide_resolution (float): Resolution of the slide.
            polygons (List): List of Shapely polygons corresponding to the Whole Slide Image object.

        Returns:
            The original slide patch, labeled mask patch, and generated prompts.
            
        """

        # List for polygons to be used in progress image
        progress_polygons = []
        
        # Determine X and Y top corner coordinates for patch
        X = int(centroid.x*slide_resolution) - 0.5*self.patch_size
        Y = int(centroid.y*slide_resolution) - 0.5*self.patch_size
        
        # Retrieve slide patch from whole slide
        slide_patch = np.array(slide.read_region((int(X/slide_resolution),int(Y/slide_resolution)), self.level, (self.patch_size,self.patch_size)).convert("RGB"))
        slide_patch_resized = self.resize.apply_image(slide_patch)
        slide_patch_np = slide_patch_resized.transpose(2,0,1)[None, :, :, :]
        
        # Find all the polygons that are in the selected patch
        patch_polygon = create_patch_polygon(X, Y, slide_resolution, self.patch_size)
        polygons_in_patch = find_polygons_patch(polygons, patch_polygon, shuffle=True)
        
        # Select 32 polygon that are in the patch to be in the progress image
        progress_polygons.append(polygon)
        for poly in polygons_in_patch:
            # Create a binary mask to check if a prompt can be created, if not skip the polygon
            binary_mask = self._draw_mask_from_poly(X, Y, poly)
            binary_mask = binary_mask.transpose(2,0,1)[None, :, :, :]
            if np.count_nonzero(binary_mask) < self.minimum_mask_size or poly['label'] == "background":
                continue
            # If not skipped then add the polygon to the list of polygons for progress image
            progress_polygons.append(poly)

            # Stop the loop if 32 polygons have been selected
            if len(progress_polygons) == 32: 
                break
        
        prompt_masks = []
        mask_patch_np = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        # Create the prompts and the gt mask
        for poly in progress_polygons:
            # Draw and save binary mask in list to create prompts
            binary_mask = self._draw_mask_from_poly(X, Y, poly)
            binary_mask = binary_mask.transpose(2,0,1)[None, :, :, :]
            prompt_masks.append(binary_mask)
            
            # Draw the polygon in the labeled gt mask patch
            label_value = self.classes.index(poly['label']) + 1
            scaled_coordinates = [[int((x / (2 ** self.level)) - X), int((y / (2 ** self.level)) - Y)] for x, y in poly["polygon"].exterior.coords]
            scaled_coordinates.append(scaled_coordinates[0])
            array_coordinates = [np.array(scaled_coordinates)]
            cv2.drawContours(mask_patch_np, array_coordinates, -1, label_value, thickness=cv2.FILLED)
        
        # Add channel and batch dimension to the gt labeled mask
        mask_patch_np = mask_patch_np[None, None,:, :] 
        
        # Generate prompts
        prompt_masks = np.concatenate(prompt_masks, axis=0)
        prompts = generate_prompts(torch.as_tensor(prompt_masks).int()).numpy()

        return slide_patch_np, mask_patch_np, prompts
        
    def _draw_mask_from_poly(self, X, Y, polygon):
        """
        Draw a binary mask from a Shapely polygon.

        Args:
            X (int): X-coordinate of the top-left corner of the patch.
            Y (int): Y-coordinate of the top-left corner of the patch.
            polygon (dict): Dictionary containing the Shapely polygon under the key 'polygon'.

        Returns:
            np.ndarray: Binary mask representing the region inside the polygon.
            
        """
        # Initialize an empty mask patch
        mask_patch_np = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        
        # Rescale and convert the polygon coordinates for OpenCV drawcontours
        scaled_coordinates = [[int((x / (2 ** self.level)) - X), int((y / (2 ** self.level)) - Y)] for x, y in polygon["polygon"].exterior.coords]
        scaled_coordinates.append(scaled_coordinates[0])
        array_coordinates = [np.array(scaled_coordinates)]
        
        # Draw filled contours on the mask patch based on the polygon
        cv2.drawContours(mask_patch_np, array_coordinates, -1, 1, thickness=cv2.FILLED)
        mask_patch_np = mask_patch_np[:, :, None]

        return mask_patch_np

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        sample_num = sample['idx']

        folder_path = sample["folder path"]
        
        input_np = tifffile.imread(folder_path / f"{sample_num}_image.tiff")
        gt_mask_np = tifffile.imread(folder_path / f"{sample_num}_gt_mask.tiff")
        gt_label_np = tifffile.imread(folder_path / f"{sample_num}_gt_label.tiff")
        
        if sample["augmentation"] == True: 
            seed = np.random.rand(3)

            if seed[0] >= 0.5 or (seed[0] < 0.5 and seed[1]< 0.5 and seed[2]>= 0.5):
                spatial_transformed = self.spatial_transform(image=np.squeeze(input_np).transpose(1,2,0), mask=np.squeeze(gt_mask_np))
                input_np = spatial_transformed['image'].transpose(2,0,1)[None,]
                gt_mask_np = spatial_transformed['mask'][None, None,]

            if seed[1] > 0.5 or (seed[0] < 0.5 and seed[1]< 0.5 and seed[2]< 0.5):
                color_transformed = self.color_transform(image=np.squeeze(input_np).transpose(1,2,0), mask=np.squeeze(gt_mask_np))
                input_np = color_transformed['image'].transpose(2,0,1)[None,]
                gt_mask_np = color_transformed['mask'][None, None,]
        
        input = torch.squeeze(torch.as_tensor(input_np), dim=0) 
        gt_mask = torch.squeeze(torch.as_tensor(gt_mask_np), dim=0).int()
        gt_label = torch.squeeze(torch.as_tensor(gt_label_np), dim=0).int()
        
        return input, gt_mask, gt_label
    