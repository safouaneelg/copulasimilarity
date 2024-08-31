"""
This implementation of the ISSM similarity metric is inspired by:
https://github.com/nekhtiari/image-similarity-measures/blob/master/image_similarity_measures/quality_metrics.py
"""

import numpy as np
from skimage.metrics import structural_similarity
import cv2
import math

class ISSMsimilarity:
    def __init__(self, max_p: int = 4095):
        """
        Initialize the ISSM similarity class.
        
        Parameters:
        - max_p: The maximum possible pixel value. Default is 255 for standard images.
        """
        self.max_p = max_p

    def ssim(self, org_img: np.ndarray, pred_img: np.ndarray) -> float:
        """
        Compute the Structural Similarity Index (SSIM) between two images.
        
        Parameters:
        - org_img: The original image.
        - pred_img: The image to compare.
        
        Returns:
        - SSIM value between the two images.
        """
        return structural_similarity(org_img, pred_img, data_range=self.max_p, channel_axis=2)

    def canny_edge_detection(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Canny Edge Detection to an image.
        
        Parameters:
        - img: The input image.
        
        Returns:
        - Edge-detected image.
        """
        return cv2.Canny(img, 100, 200)

    def ehs(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Entropy-Histogram Similarity measure.
        
        Parameters:
        - x: The first image.
        - y: The second image.
        
        Returns:
        - Entropy-Histogram Similarity value.
        """
        if np.array_equal(x, y):
            return 0  # Return 0 when images are identical
        
        H = np.histogram2d(x.flatten(), y.flatten(), bins=255, range=[[0, 255], [0, 255]])[0]
        H += 1e-10  # Add small value to avoid log(0)
        H /= np.sum(H)  # Normalize

        return -np.sum(H * np.log2(H))

    def edge_c(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Edge Correlation coefficient based on Canny detection.
        
        Parameters:
        - x: The first image.
        - y: The second image.
        
        Returns:
        - Edge Correlation value.
        """
        if np.array_equal(x, y):
            return 1  # Return 1 when images are identical
        
        g = self.canny_edge_detection(x)
        h = self.canny_edge_detection(y)

        g0 = np.mean(g)
        h0 = np.mean(h)

        numerator = np.sum((g - g0) * (h - h0))
        denominator = np.sqrt(np.sum(np.square(g - g0)) * np.sum(np.square(h - h0)))

        return numerator / denominator if denominator != 0 else 0

    def issm(self, org_img: np.ndarray, pred_img: np.ndarray) -> float:
        """
        Compute the Information theoretic-based Statistic Similarity Measure (ISSM).
        
        Parameters:
        - org_img: The original image.
        - pred_img: The image to compare.
        
        Returns:
        - ISSM value between the two images.
        """
        if np.array_equal(org_img, pred_img):
            return 1  # Return 1 when images are identical

        ehs_val = self.ehs(org_img, pred_img)
        canny_val = self.edge_c(org_img, pred_img)
        ssim_val = self.ssim(org_img, pred_img)

        A = 0.3
        B = 0.5
        C = 0.7

        denominator = A * canny_val * ehs_val + B * ehs_val + C * ssim_val

        # Avoid division by zero
        if denominator == 0:
            return np.nan

        return (canny_val * ehs_val * (A + B)) / denominator
