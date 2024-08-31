"""
This implementation of the FSIM similarity metric is inspired by:
https://github.com/nekhtiari/image-similarity-measures/blob/master/image_similarity_measures/quality_metrics.py
"""

import numpy as np
import cv2
import phasepack.phasecong as pc

class FSIMsimilarity:
    def __init__(self, T1: float = 0.85, T2: float = 160):
        """
        Initialize the FSIM similarity class.

        Parameters:
        - T1: Constant based on the dynamic range of phase congruency values. Default is 0.85.
        - T2: Constant based on the dynamic range of gradient magnitude values. Default is 160.
        """
        self.T1 = T1
        self.T2 = T2

    def _similarity_measure(self, x: np.ndarray, y: np.ndarray, constant: float) -> np.ndarray:
        """
        Calculate feature similarity measurement between two images.

        Parameters:
        - x: First image.
        - y: Second image.
        - constant: Constant to be used in the similarity calculation.

        Returns:
        - Similarity measure between the two images.
        """
        numerator = 2 * np.multiply(x, y) + constant
        denominator = np.add(np.square(x), np.square(y)) + constant

        return np.divide(numerator, denominator)

    def _gradient_magnitude(self, img: np.ndarray, img_depth: int) -> np.ndarray:
        """
        Calculate gradient magnitude based on Scharr operator.

        Parameters:
        - img: Input image.
        - img_depth: Depth of the image data type.

        Returns:
        - Gradient magnitude image.
        """
        scharrx = cv2.Scharr(img, img_depth, 1, 0)
        scharry = cv2.Scharr(img, img_depth, 0, 1)

        return np.sqrt(scharrx**2 + scharry**2)

    def fsim(self, org_img: np.ndarray, pred_img: np.ndarray) -> float:
        """
        Compute the Feature-based Similarity Index (FSIM) between two RGB images.

        Parameters:
        - org_img: The original RGB image.
        - pred_img: The predicted RGB image.

        Returns:
        - FSIM value between the two images.
        """
        if org_img.ndim == 2:
            org_img = np.expand_dims(org_img, axis=-1)
        if pred_img.ndim == 2:
            pred_img = np.expand_dims(pred_img, axis=-1)

        alpha = beta = 1  # Parameters to adjust the relative importance of PC and GM features
        fsim_list = []

        for i in range(org_img.shape[2]):
            # Calculate the Phase Congruency (PC) for original and predicted images
            pc1_2dim = pc(
                org_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978
            )
            pc2_2dim = pc(
                pred_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978
            )

            # Sum up the PC values over all orientations
            pc1_2dim_sum = np.sum(pc1_2dim[4], axis=0)
            pc2_2dim_sum = np.sum(pc2_2dim[4], axis=0)

            # Calculate Gradient Magnitude (GM) for original and predicted images
            gm1 = self._gradient_magnitude(org_img[:, :, i], cv2.CV_16U)
            gm2 = self._gradient_magnitude(pred_img[:, :, i], cv2.CV_16U)

            # Calculate similarity measures for PC and GM
            S_pc = self._similarity_measure(pc1_2dim_sum, pc2_2dim_sum, self.T1)
            S_g = self._similarity_measure(gm1, gm2, self.T2)

            # Compute the FSIM value for the current channel
            S_l = (S_pc**alpha) * (S_g**beta)
            numerator = np.sum(S_l * np.maximum(pc1_2dim_sum, pc2_2dim_sum))
            denominator = np.sum(np.maximum(pc1_2dim_sum, pc2_2dim_sum))
            fsim_list.append(numerator / denominator)

        return np.mean(fsim_list)
