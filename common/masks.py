import numpy as np
import cv2
import skimage
from skimage.measure import label

def generate_tissue_mask(slide):
        """Creates a mask out of a WSI by applying a Gaussian filter to blur the image and consequently apply Otsu's thresholding to it

        Args:
            slide (OpenSlide): an OpenSlide object

        Returns:
            mask: a 2D map of the mask
            pixels: non-zero coordinates of the mask
            downsample
        """
        # Thresholding
        dlevel = len(slide.level_dimensions) - 3
        downsample = slide.level_downsamples[dlevel]
        overview = slide.read_region((0, 0), dlevel, slide.level_dimensions[dlevel])
        img = np.array(overview)
        black_pixels = np.where(
            (img[:, :, 0] <50) & 
            (img[:, :, 1] <50) & 
            (img[:, :, 2] <50)
        )

        # set those pixels to white
        img[black_pixels] = [240, 240, 240, 0]
        img = img[:, :, ::-1].copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.bitwise_not(th3)

        # Remove small islands
        labelMap = label(mask)
        icc = len(np.unique(labelMap))
        for i in range(icc):
            if (
                np.sum(labelMap == i) < 500
            ):  # cluster with a number of pixels smaller than this threshold will be set to zero
                mask[labelMap == i] = 0

        # Find pixels
        disk = skimage.morphology.disk(7) 
        mask = skimage.morphology.binary_dilation(mask, footprint=disk)
        mask = mask.transpose(1,0) 

        pixels = np.nonzero(mask)
        return mask, pixels