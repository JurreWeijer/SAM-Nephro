import sys
sys.path.append("..")

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from functools import partial
import copy

from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, TwoWayTransformer
from fine_tuning_classification.class_decoder import MaskClassDecoder


def build_samclass_vit_h(classes, freeze_image_encoder, freeze_prompt_encoder, freeze_mask_decoder, checkpoint=None):
    return _build_sam_class(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        classes=classes,
        freeze_image_encoder=freeze_image_encoder,
        freeze_prompt_encoder=freeze_prompt_encoder,
        freeze_mask_decoder=freeze_mask_decoder,
        checkpoint=checkpoint,
    )

def build_samclass_vit_l(classes, freeze_image_encoder, freeze_prompt_encoder, freeze_mask_decoder, checkpoint=None):
    return _build_sam_class(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        classes=classes,
        freeze_image_encoder=freeze_image_encoder,
        freeze_prompt_encoder=freeze_prompt_encoder,
        freeze_mask_decoder=freeze_mask_decoder,
        checkpoint=checkpoint,
    )

def build_samclass_vit_b(classes, freeze_image_encoder, freeze_prompt_encoder, freeze_mask_decoder, checkpoint=None):
    return _build_sam_class(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        classes=classes,
        freeze_image_encoder=freeze_image_encoder,
        freeze_prompt_encoder=freeze_prompt_encoder,
        freeze_mask_decoder=freeze_mask_decoder,
        checkpoint=checkpoint,
    )

sam_model_registry = {
    "default": build_samclasssingle_vit_h,
    "vit_h": build_samclasssingle_vit_h,
    "vit_l": build_samclasssingle_vit_l,
    "vit_b": build_samclasssingle_vit_b,
}

def _build_sam_class(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    classes,
    freeze_image_encoder,
    freeze_prompt_encoder,
    freeze_mask_decoder,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = SamClass(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskClassDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes=len(classes)
        ),
        freeze_image_encoder=freeze_image_encoder,
        freeze_prompt_encoder=freeze_prompt_encoder,
        freeze_mask_decoder=freeze_mask_decoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device('cpu'))
        
        class_decoder_weights = {k.replace("class_decoder.", ""): v for k, v in state_dict.items() if k.startswith('class_decoder')}
        if len(class_decoder_weights) > 0:
            sam.load_state_dict(state_dict)
        else:
        
            image_encoder_weights = {k.replace("image_encoder.", ""): v for k, v in state_dict.items() if k.startswith('image_encoder')}
            prompt_encoder_weights = {k.replace("prompt_encoder.", ""): v for k, v in state_dict.items() if k.startswith('prompt_encoder')}
            mask_decoder_weights = {k.replace("mask_decoder.", ""): v for k, v in state_dict.items() if k.startswith('mask_decoder')}        
            cnn_decoder_weights = {k.replace("cnn_encoder.", ""): v for k, v in state_dict.items() if k.startswith('cnn_encoder')}
            
            #mask_decoder_weights["mask_tokens.weight"] = mask_decoder_weights["mask_tokens.weight"][0:1,:]
            mask_decoder_weights["mask_tokens.weight"] = mask_decoder_weights["mask_tokens.weight"][-1:,:]
            mask_decoder_weights["iou_prediction_head.layers.2.weight"] = mask_decoder_weights["iou_prediction_head.layers.2.weight"][0:1,:]
            mask_decoder_weights["iou_prediction_head.layers.2.bias"] = mask_decoder_weights["iou_prediction_head.layers.2.bias"][0:1]

            sam.image_encoder.load_state_dict(image_encoder_weights)
            sam.prompt_encoder.load_state_dict(prompt_encoder_weights)
            sam.mask_decoder.load_state_dict(mask_decoder_weights, strict=False)
            
            if sam.cnn_encoder is not None and len(cnn_decoder_weights) > 0: 
                sam.cnn_encoder.load_state_dict(cnn_decoder_weights)

    return sam


##################################################################################################################################################
##################################################################################################################################################

class SamClass(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskClassDecoder,
        cnn_encoder: nn.Module = None,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = False,
        freeze_mask_decoder = False, 
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction. 
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          cnn_encoder: second encoder to supply task specific information to the model.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.cnn_encoder = cnn_encoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
        self.freeze_image_encoder = freeze_image_encoder
        self.freeze_prompt_encoder = freeze_prompt_encoder
        self.freeze_mask_decoder = freeze_mask_decoder
        self.freeze_class_decoder = False
        
        if self.cnn_encoder is not None:
            self.alpha = nn.Parameter(torch.zeros(1))
            self.Sigmoid = nn.Sigmoid()
        
        for n, p in self.mask_decoder.named_parameters():
            if freeze_mask_decoder == "segmentation":
                if not "cls" in n and not "class" in n: 
                    p.requires_grad = False 
            elif freeze_mask_decoder == 'classification':
                if "cls" in n or "class" in n: 
                    p.requires_grad = False
            
        if freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        if freeze_prompt_encoder:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
        if freeze_mask_decoder == True:
            for param in self.mask_decoder.parameters():
                param.requires_grad = False
                    
    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    def forward(self, input, prompts):
        # Only for training, for inference use the predictor
        
        if input.shape[-1] == 1024:
            preprocesed_input = self.preprocess(input)
            image_embeddings = self.image_encoder(preprocesed_input)
        else:
            image_embeddings = input
        
        predicted_masks = []
        predicted_classes = []
        predicted_iou = []
        for image_embedding, prompt in zip(image_embeddings, prompts):
            image_embedding = image_embedding[None,:,:,:]
            
            point_labels = torch.ones(prompt.shape[:-1])
            points = (prompt, point_labels)

            #forward pass prompt encoder, without gradient to freeze the weights
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            ) 

            low_res_mask, class_prediction, pred_iou = self.mask_decoder(
                image_embeddings=image_embedding,
                image_pe= self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            predicted_masks.append(low_res_mask)
            predicted_classes.append(class_prediction)
            predicted_iou.append(pred_iou)
        
        predicted_masks = torch.cat(predicted_masks, dim=0)
        predicted_classes = torch.cat(predicted_classes, dim = 0)
        predicted_iou = torch.cat(predicted_iou, dim=0)
           
        return predicted_masks, predicted_classes, predicted_iou

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
