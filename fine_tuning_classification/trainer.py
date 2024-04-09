import sys
sys.path.append("/hpc/dla_patho/home/jweijer/Internship-Jurre")

import torch
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import random
import json
import os
     
###############################################################################################################################################
###############################################################################################################################################
            
class ClassificationTrainer():
    def __init__(
        self,
        model,
        segmentation_loss,
        classification_loss,
        iou_loss,
        segmentation_weight,
        classification_weight,
        iou_weight,
        optimizer,
        train_loader,
        valid_loader,
        classes, 
        accumulation_steps, 
        progress_dir,
        seed,
        warmup_period = None,
        lr_decay = None, 
        tolerance = 0.01,
        save_progress = True,
        progress_images = True,
        display_freq = 5,
        early_stopping = False,
    ):
        """
        Trainer class for a deep learning model with segmentation and classification tasks.

        Args:
            model: The PyTorch model to be trained.
            segmentation_loss: The loss function for segmentation task.
            classification_loss: The loss function for classification task.
            iou_loss: The loss function for IoU calculation.
            segmentation_weight: Weight for the segmentation loss in the total loss.
            classification_weight: Weight for the classification loss in the total loss.
            iou_weight: Weight for the IoU loss in the total loss.
            optimizer: The optimizer used for training the model.
            train_loader: DataLoader for the training dataset.
            valid_loader: DataLoader for the validation dataset.
            classes: Number of classes in the classification task.
            accumulation_steps: Number of steps to accumulate gradients before updating weights, if set to zero no gradient accumulation is used.
            progress_dir: Directory to save training progress.
            seed: Random seed for reproducibility.
            warmup_period: Number of warm-up iterations for learning rate.
            lr_decay: Learning rate decay factor.
            tolerance: Tolerance for early stopping.
            save_progress: Flag to save training progress.
            progress_images: Flag to save intermediate progress images during training.
            display_freq: Frequency of saving/displaying progress images.
            early_stopping: Flag to enable early stopping during training.
        
        """
        self.model = model
        self.segmentation_loss = segmentation_loss
        self.classification_loss = classification_loss
        self.iou_loss = iou_loss
        self.segmentation_weight = segmentation_weight
        self.classification_weight = classification_weight
        self.iou_weight = iou_weight
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.classes = classes
        self.accumulation_steps = accumulation_steps
        self.progress_dir = progress_dir
        self.seed = seed
        self.tolerance = tolerance
        self.save_progress = save_progress
        self.progress_images = progress_images
        self.display_freq = display_freq
        self.early_stopping = early_stopping
        self.epochs_run = 0
        
        self.lr_decay = lr_decay
        self.warmup_period = warmup_period
        self.base_lr =  self.optimizer.param_groups[0]['lr'] * self.warmup_period
        
        self.minimum_valid_loss = 1e10
        self.train_losses = []
        self.train_segmentation_losses = []
        self.train_classification_losses = []
        self.train_iou_losses = []
        self.train_pred_ious = []
        self.train_stability_scores = []
        
        self.valid_losses = []
        self.valid_segmentation_losses = []
        self.valid_classification_losses = []
        self.valid_iou_losses = []
        self.valid_pred_ious = []
        self.valid_stability_scores = []
        
        snapshot_path = self.progress_dir / "snapshot.pt"
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
            
        #set the random seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        if self.progress_images == True:
            self.image_gt_t = torch.as_tensor(train_loader.dataset.progress_images)
            self.mask_gt_t = torch.as_tensor(train_loader.dataset.progress_masks)
            self.labels_gt_t = torch.as_tensor(train_loader.dataset.progress_prompts)
            
            self.image_gt_v = torch.as_tensor(valid_loader.dataset.progress_images)
            self.mask_gt_v = torch.as_tensor(valid_loader.dataset.progress_masks)
            self.label_gt_v = torch.as_tensor(valid_loader.dataset.progress_prompts)
            
        print(f"trainer device: {self.device}")
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return self.model.device

    def _load_snapshot(self, snapshot_path):
        """Load model and training state from a snapshot."""
        snapshot = torch.load(snapshot_path, map_location=self.device)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"] 
        self.train_losses = snapshot["TRAIN_LOSS"]
        self.train_segmentation_losses = snapshot["TRAIN_INSTANCE_LOSS"]
        self.train_classification_losses = snapshot["TRAIN_CLASSSIFICATION_LOSS"]
        #self.train_iou_losses = snapshot["TRAIN_IOU_LOSS"]
        #self.train_pred_ious = snapshot["TRAIN_PRED_IOUS"]
        self.train_stability_scores = snapshot["TRAIN_STABILITY_SCORES"]
        self.valid_losses = snapshot["VALID_LOSS"]
        self.valid_segmentation_losses = snapshot["VALID_INSTANCE_LOSS"]
        self.valid_classification_losses = snapshot["VALID_CLASSIFICATION_LOSS"]
        #self.valid_iou_losses = snapshot["VALID_IOU_LOSS"]
        #self.valid_pred_ious = snapshot["VALID_PRED_IOUS"]
        self.valid_stability_scores = snapshot["VALID_STABILITY_SCORE"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run + 1}")
    
    
    def _forward_model(self, inputs, prompts):
        """Forward pass through the model."""
        
        if inputs.shape[-1] == 1024:
            preprocesed_input = self.model.preprocess(inputs)
            image_embeddings = self.model.image_encoder(preprocesed_input)
        else:
            image_embeddings = inputs
        
        if self.model.cnn_encoder is not None:
            gate = self.model.Sigmoid(self.model.alpha)
            cnn_embedding = self.model.cnn_encoder(preprocesed_input)
            image_embeddings = gate*image_embeddings + (1-gate) * cnn_embedding
        
        predicted_masks = []
        predicted_classes = []
        predicted_iou = []
        for image_embedding, prompt in zip(image_embeddings, prompts):
            image_embedding = image_embedding[None,:,:,:]
            
            point_labels = torch.ones(prompt.shape[:-1])
            points = (prompt, point_labels)

            #forward pass prompt encoder, without gradient to freeze the weights
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            ) 

            low_res_mask, class_prediction, pred_iou = self.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe= self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            predicted_masks.append(low_res_mask)
            predicted_classes.append(class_prediction)
            predicted_iou.append(pred_iou)
        
        predicted_masks = torch.cat(predicted_masks, dim=0)
        predicted_classes = torch.cat(predicted_classes, dim=0)
        predicted_iou = torch.cat(predicted_iou, dim=0)
           
        return predicted_masks, predicted_classes, predicted_iou
        
    def _run_batch(self, inputs, masks, labels, prompts):
        """Run a training batch and calculate losses."""
        
        predicted_masks, predicted_classes, pred_iou = self._forward_model(inputs, prompts)
            
        scaled_masks = F.interpolate(masks.float(), tuple(predicted_masks.shape[-2:]), mode='nearest')
        segmentation_loss = self.segmentation_loss(predicted_masks, scaled_masks)
        
        tresholded_mask = torch.where(predicted_masks > self.model.mask_threshold, 1, 0)
        true_iou = calculate_iou(tresholded_mask, scaled_masks)
        iou_loss = self.iou_loss(pred_iou, true_iou)
        
        classification_loss = self.classification_loss(predicted_classes, labels.float())
        
        loss_value = self.segmentation_weight*segmentation_loss + self.classification_weight*classification_loss + self.iou_weight*iou_loss

        post_processed_masks = self.model.postprocess_masks(predicted_masks, (1024, 1024), (1024,1024))
        stability_scores = calculate_stability_score(post_processed_masks, self.model.mask_threshold, 1.0)
            
        return loss_value, segmentation_loss, classification_loss, iou_loss, pred_iou, stability_scores
        
    def _train_epoch(self):
        """Run a training epoch."""
        
        train_epoch_losses = 0
        train_segmentation_losses = 0
        train_classification_losses = 0
        train_iou_losses = 0
        train_stability_scores = []
        train_pred_ious = []
        self.model.train()
        for step, (inputs, masks, labels) in enumerate(tqdm(self.train_loader), start=1):
            prompts = generate_prompts(masks)
            if prompts is None:
                continue
              
            inputs, masks, prompts, labels = inputs.to(self.device), masks.to(self.device), prompts.to(self.device), labels.to(self.device)
            
            loss_value, segmentation_loss, classification_loss, iou_loss, pred_iou, stability_scores = self._run_batch(inputs, masks, labels, prompts)
            loss_value.backward()
            
            if self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                if (step+1) % self.accumulation_steps == 0:
                    for param in self.model.parameters():
                        if param.requires_grad and param.grad is not None: 
                            param.grad /= self.accumulation_steps
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            if self.warmup_period > 0 and self.iter_num < self.warmup_period:
                lr = self.base_lr * ((self.iter_num + 1) / self.warmup_period)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            elif self.lr_decay:
                shift_iter = self.iter_num - self.warmup_period
                lr = self.base_lr * (1.0 - shift_iter / self.max_iterations) ** 0.9
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
            train_epoch_losses += loss_value.detach().cpu().item()
            train_segmentation_losses += segmentation_loss.detach().cpu().item()
            train_classification_losses += classification_loss.detach().cpu().item()
            train_iou_losses += iou_loss.detach().cpu().item()
            self.iter_num += 1
            for iou, stability in zip(pred_iou, stability_scores):
                train_pred_ious.append(iou.detach().cpu().item())
                train_stability_scores.append(stability.detach().cpu().item())      
        
        inputs, masks, prompts, labels = inputs.detach().cpu(), masks.detach().cpu(), prompts.detach().cpu(), labels.detach().cpu()
                
        train_epoch_loss = train_epoch_losses/step
        train_segmentation_loss = train_segmentation_losses/step
        train_classification_loss = train_classification_losses/step
        train_iou_loss = train_iou_losses/step
        
        self.train_pred_ious.append(np.array(train_pred_ious)[None,])
        self.train_stability_scores.append(np.array(train_stability_scores)[None,])
        
        return train_epoch_loss, train_segmentation_loss, train_classification_loss, train_iou_loss
    
    def _valid_epoch(self):
        """Run a validation epoch."""
        
        valid_epoch_losses = 0
        valid_segmentation_losses = 0
        valid_classification_losses = 0
        valid_iou_losses = 0
        valid_stability_scores = []
        valid_pred_ious = []
        self.model.train()
        for step, (inputs, masks, labels) in enumerate(tqdm(self.valid_loader), start=1):
            prompts = generate_prompts(masks)
            if prompts is None:
                continue
            
            inputs, masks, prompts, labels = inputs.to(self.device), masks.to(self.device), prompts.to(self.device), labels.to(self.device)
            loss_value, segmentation_loss, classification_loss, iou_loss, pred_iou, stability_scores = self._run_batch(inputs, masks, labels, prompts)
            
            valid_epoch_losses += loss_value.detach().cpu().item()
            valid_segmentation_losses += segmentation_loss.detach().cpu().item()
            valid_classification_losses += classification_loss.detach().cpu().item()
            valid_iou_losses += iou_loss.detach().cpu().item()
            
            for iou, stability in zip(pred_iou, stability_scores):
                valid_pred_ious.append(iou.detach().cpu().item())
                valid_stability_scores.append(stability.detach().cpu().item())
        
        inputs, masks, prompts, labels = inputs.detach().cpu(), masks.detach().cpu(), prompts.detach().cpu(), labels.detach().cpu()
        
        valid_epoch_loss = valid_epoch_losses/step
        valid_segmentation_loss = valid_segmentation_losses/step
        valid_classification_loss = valid_classification_losses/step
        valid_iou_loss = valid_iou_losses/step
        
        self.valid_pred_ious.append(np.array(valid_pred_ious)[None,])
        self.valid_stability_scores.append(np.array(valid_stability_scores)[None,])
        
        return valid_epoch_loss, valid_segmentation_loss, valid_classification_loss, valid_iou_loss
     
    def _progress_inference(self, progress_images, progress_masks, progress_prompts):
        """Run inference on progress images."""
        
        progress_images = progress_images.to(self.device)
        progress_masks = progress_masks.to(self.device)
        progress_prompts = progress_prompts.to(self.device)
        
        output_predictions = []
        for image, mask, prompt in zip(progress_images, progress_masks, progress_prompts):
            image, mask, prompt = image[None,], mask[None], prompt[None,]
            
            predicted_masks, predicted_classes, _ = self._forward_model(image, prompt)
            resized_predicted_masks = F.interpolate(predicted_masks.float(), tuple(mask.shape[-2:]), mode='nearest')
            
            output_prediction = torch.zeros_like(mask)
            for pred_mask, pred_class in zip(resized_predicted_masks, predicted_classes):
                pred_mask = pred_mask[None,]
                class_label = torch.argmax(pred_class) + 1

                output_prediction = torch.where(pred_mask > self.model.mask_threshold, class_label, output_prediction)
                
            output_predictions.append(output_prediction)
        
        output_predictions = torch.cat(output_predictions, axis=0)
        
        return output_predictions.float(), progress_masks.float()  
    
    def _save_progress_images(self, epoch):
        """Save progress images during training."""
        
        if epoch% self.display_freq == 0:
            self.model.eval()
            
            #run inference on the progress image using curren state of the model
            image_pred_t, image_gt_t = self._progress_inference(self.image_gt_t, self.mask_gt_t, self.labels_gt_t)
            image_pred_v, image_gt_v = self._progress_inference(self.image_gt_v, self.mask_gt_v, self.label_gt_v)
            
            #combine progress images in a grid
            image_grid = make_grid(
                    torch.cat([
                        image_gt_t.cpu(), 
                        image_pred_t.cpu(),
                        image_gt_v.cpu(), 
                        image_pred_v.cpu(), 
                    ]), 
                        nrow=image_gt_t.shape[0], 
                        padding=12, 
                        pad_value=-5, 
                    )
            image_grid = image_grid.numpy()[0]
            plt.imsave(self.progress_dir / f"progress_image_{epoch:03d}.png",  image_grid) 
    
    def _save_snapshot(self, epoch):
        """Save a snapshot of the model and training state."""
        
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "EPOCHS_RUN": epoch,
            "TRAIN_LOSS": self.train_losses,
            "TRAIN_INSTANCE_LOSS": self.train_segmentation_losses,
            "TRAIN_CLASSSIFICATION_LOSS": self.train_classification_losses,
            "TRAIN_IOU_LOSS": self.train_iou_losses,
            "TRAIN_PRED_IOUS": self.train_pred_ious,
            "TRAIN_STABILITY_SCORES": self.train_stability_scores,
            "VALID_LOSS": self.valid_losses,
            "VALID_INSTANCE_LOSS": self.valid_segmentation_losses,
            "VALID_CLASSIFICATION_LOSS": self.valid_classification_losses,
            "VALID_IOU_LOSS": self.valid_iou_losses,
            "VALID_PRED_IOUS": self.valid_pred_ious,
            "VALID_STABILITY_SCORES": self.valid_stability_scores,
        }
        snapshot_path = self.progress_dir/"snapshot.pt"
        torch.save(snapshot, snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {snapshot_path}")  
    
    def _save_losses(self):
        """Save training and validation losses."""
        
        losses = {"train loss": self.train_losses,
                  "min train loss": {self.train_losses.index(min(self.train_losses)):min(self.train_losses)},
                  "valid loss": self.valid_losses,
                  "min valid loss": {self.valid_losses.index(min(self.valid_losses)):min(self.valid_losses)},
                  "train segmentation loss": self.train_segmentation_losses,
                  "min train segmentation loss": {self.train_segmentation_losses.index(min(self.train_segmentation_losses)):min(self.train_segmentation_losses)},
                  "valid segmentation loss": self.valid_segmentation_losses,
                  "min valid segmentation loss": {self.valid_segmentation_losses.index(min(self.valid_segmentation_losses)):min(self.valid_segmentation_losses)},
                  "train classification loss": self.train_classification_losses,
                  "min train classification loss": {self.train_classification_losses.index(min(self.train_classification_losses)):min(self.train_classification_losses)},
                  "valid classification loss": self.valid_classification_losses,
                  "min valid classification loss": {self.valid_classification_losses.index(min(self.valid_classification_losses)):min(self.valid_classification_losses)},
                  "train iou loss": self.train_iou_losses,
                  "min train iou loss": {self.train_iou_losses.index(min(self.train_iou_losses)):min(self.train_iou_losses)},
                  "valid iou loss": self.valid_iou_losses,
                  "min valid iou loss": {self.valid_iou_losses.index(min(self.valid_iou_losses)):min(self.valid_iou_losses)},
                } 
    
        with open(self.progress_dir / 'losses.json', 'w') as j:
                json.dump(losses, j)
    
    def _save_loss_image(self):
        """Save an image visualizing training and validation losses."""
        
        fig, axes = plt.subplots(1, 4, figsize=(25, 5))
            
        # Subplot 1 - Train and Validation Loss
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.valid_losses, label='Validation Loss')
        axes[0].set_title("Total Loss")
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        # Subplot 2 - Train and Validation Loss (duplicate for the second set of values)
        axes[1].plot(self.train_segmentation_losses, label='Train Loss')
        axes[1].plot(self.valid_segmentation_losses, label='Validation Loss')
        axes[1].set_title("Segmentation Loss")
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()

        # Subplot 3 - Train and Validation Loss (duplicate for the third set of values)
        axes[2].plot(self.train_classification_losses, label='Train Loss')
        axes[2].plot(self.valid_classification_losses, label='Validation Loss')
        axes[2].set_title("Classification Loss")
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        
        # Subplot 3 - Train and Validation Loss (duplicate for the third set of values)
        axes[3].plot(self.train_iou_losses, label='Train Loss')
        axes[3].plot(self.valid_iou_losses, label='Validation Loss')
        axes[3].set_title("Iou Loss")
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Loss')
        axes[3].legend()

        # Adjust layout to prevent subplot overlapping
        plt.tight_layout()
        plt.savefig(self.progress_dir / "loss_image.png")
        plt.close()    
        
    def _save_score_image(self):
        """Save an image visualizing mean and std for pred_iou and stability_score."""
        
        # Convert lists of lists to NumPy arrays
        #train_iou_array = np.concatenate(self.train_pred_ious, axis=0)
        #validation_iou_array = np.concatenate(self.valid_pred_ious, axis=0)
        #train_stability_array = np.concatenate(self.train_stability_scores, axis=0)
        #validation_stability_array = np.concatenate(self.valid_stability_scores, axis=0)

        # Calculate mean and standard deviation for each epoch
        #mean_train_iou = np.mean(train_iou_array, axis=1)
        #std_train_iou = np.std(train_iou_array, axis=1)
        #mean_train_stability = np.mean(train_stability_array, axis=1)
        #std_train_stability = np.std(train_stability_array, axis=1)

        #mean_validation_iou = np.mean(validation_iou_array, axis=1)
        #std_validation_iou = np.std(validation_iou_array, axis=1)
        #mean_validation_stability = np.mean(validation_stability_array, axis=1)
        #std_validation_stability = np.std(validation_stability_array, axis=1)
        
        mean_train_iou = [np.mean(array) for array in self.train_pred_ious]
        std_train_iou = [np.std(array) for array in self.train_pred_ious]
        mean_train_stability = [np.mean(array) for array in self.train_stability_scores]
        std_train_stability = [np.std(array) for array in self.train_stability_scores]

        mean_validation_iou = [np.mean(array) for array in self.valid_pred_ious]
        std_validation_iou = [np.std(array) for array in self.valid_pred_ious]
        mean_validation_stability = [np.mean(array) for array in self.valid_stability_scores]
        std_validation_stability = [np.std(array) for array in self.valid_stability_scores]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Subplot 1 - Train Mean and Std for pred_iou
        axes[0, 0].errorbar(range(1, len(mean_train_iou) + 1), mean_train_iou, yerr=std_train_iou, label='Train')
        axes[0, 0].set_title("Train Mean and Std for pred_iou")
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('pred_iou')
        #axes[0, 0].set_ylim([0.25, 1])
        axes[0, 0].legend()

        # Subplot 2 - Validation Mean and Std for pred_iou
        axes[0, 1].errorbar(range(1, len(mean_validation_iou) + 1), mean_validation_iou, yerr=std_validation_iou, label='Validation')
        axes[0, 1].set_title("Validation Mean and Std for pred_iou")
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('pred_iou')
        #axes[0, 1].set_ylim([0.25, 1])
        axes[0, 1].legend()

        # Subplot 3 - Train Mean and Std for stability_score
        axes[1, 0].errorbar(range(1, len(mean_train_stability) + 1), mean_train_stability, yerr=std_train_stability, label='Train')
        axes[1, 0].set_title("Train Mean and Std for stability_score")
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('stability_score')
        #axes[1, 0].set_ylim([0.25, 1])
        axes[1, 0].legend()

        # Subplot 4 - Validation Mean and Std for stability_score
        axes[1, 1].errorbar(range(1, len(mean_validation_stability) + 1), mean_validation_stability, yerr=std_validation_stability, label='Validation')
        axes[1, 1].set_title("Validation Mean and Std for stability_score")
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('stability_score')
        #axes[1, 1].set_ylim([0.25, 1])
        axes[1, 1].legend()

        # Adjust layout to prevent subplot overlapping
        plt.tight_layout()
        plt.savefig(self.progress_dir / "scores_image.png")
        plt.close()  

    def train(self, max_epochs):
        """Train the model for a specified number of epochs."""
        
        b_sz = len(next(iter(self.train_loader))[0])
        self.max_iterations = max_epochs * (len(self.train_loader.dataset)// b_sz)
        self.iter_num = self.epochs_run * (len(self.train_loader.dataset)// b_sz)
        
        for epoch in range(self.epochs_run + 1, max_epochs+1):
            print(f"starting epoch: {epoch}")
            # Run train epoch 
            train_epoch_loss, train_segmentation_loss, train_classification_loss, train_iou_loss = self._train_epoch()
            self.train_losses.append(train_epoch_loss)
            self.train_segmentation_losses.append(train_segmentation_loss)
            self.train_classification_losses.append(train_classification_loss)
            self.train_iou_losses.append(train_iou_loss)
            
            # Run train epoch  
            valid_epoch_loss, valid_segmentation_loss, valid_classification_loss, valid_iou_loss = self._valid_epoch()
            self.valid_losses.append(valid_epoch_loss)
            self.valid_segmentation_losses.append(valid_segmentation_loss)
            self.valid_classification_losses.append(valid_classification_loss)
            self.valid_iou_losses.append(valid_iou_loss)
            
            # Print progress
            print(f'Epoch #{epoch:03d}: Train Loss: {train_epoch_loss:.3f}, Train Segmentation Loss: {train_segmentation_loss:.3f}, Train Classification Loss {train_classification_loss:.3f} | Valid Loss: {valid_epoch_loss:.3f}, Valid Segmentation Loss: {valid_segmentation_loss:.3f}, Valid Classification Loss: {valid_classification_loss:.3f}')
            
            # Save snapshot
            self._save_snapshot(epoch)
            
            # Save progress
            if self.progress_images:
                self._save_progress_images(epoch) 
            
            if self.save_progress: 
                self._save_losses()
                self._save_loss_image()
                self._save_score_image()
            
            # Save the latest model 
            torch.save(self.model.state_dict(), self.progress_dir / "sam_model_latest.pth")
        
            # Save the best model
            if valid_epoch_loss < self.minimum_valid_loss + self.tolerance:
                no_increase = 0
                self.minimum_valid_loss = valid_epoch_loss
                torch.save(self.model.state_dict(), self.progress_dir / 'sam_model_best.pth')
            elif self.early_stopping:
                no_increase += 1
                if no_increase > 5:
                    break

def generate_prompts(masks):
    prompts = list()
    for mask in masks:
        mask = mask[None,]
        
        #schrink the object in the mask 
        binary_mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shrunk_mask_np = cv2.erode(binary_mask_np, kernel, iterations=1)
        shrunk_mask = torch.from_numpy(shrunk_mask_np).unsqueeze(0).unsqueeze(0)
        
        #retriev the coordinates of the pixel belonging to the schrunk object 
        object_coordinates = torch.nonzero(shrunk_mask.permute(0,1,3,2))
        if object_coordinates.size(0) <= 0:
            object_coordinates = torch.nonzero(mask.permute(0,1,3,2))
        
        if object_coordinates.size(0) > 0:
            random_index = random.randint(0, object_coordinates.size(0) - 1)
            random_point = object_coordinates[random_index]
            random_point = random_point[-2:][None, None,]
    
            prompts.append(random_point)
        else:
            return None

    
    prompts = torch.cat(prompts, dim=0)

    return prompts[None,]

def calculate_iou(pred_mask, gt_mask, smooth=1e-5):
        """Calculate Intersection over Union (IoU) between predicted and ground truth masks."""
        
        # Flatten the tensors
        pd = pred_mask.view(pred_mask.size(0), pred_mask.size(1), -1)
        gt = gt_mask.view(gt_mask.size(0), gt_mask.size(1), -1)

        # Compute intersection and union
        intersection = torch.sum(torch.min(pd, gt), dim=2)
        union = torch.sum(pd, dim=2) + torch.sum(gt, dim=2) - intersection

        # Calculate IoU
        iou = intersection / union + smooth
        
        return iou

def calculate_stability_score(
    masks: torch.Tensor, mask_threshold: float, threshold_offset: float, smooth: float = 1e-5
) -> torch.Tensor:
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # One mask is always contained inside the other.
    # Save memory by preventing unnecessary cast to torch.int64
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    unions = (
        (masks > (mask_threshold - threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )

    # Add smooth parameter to both numerator and denominator
    mask_iou = (intersections + smooth) / (unions + smooth)

    return mask_iou