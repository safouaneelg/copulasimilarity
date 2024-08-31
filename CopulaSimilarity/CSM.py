import numpy as np
import math
from scipy.ndimage import generic_filter

def erfinv(x):
    """Approximate the inverse error function using a polynomial approximation."""
    a = 0.147  # Constant used in the approximation
    
    # Compute the approximation for erfinv
    ln_part = np.log(1 - x**2)
    term1 = (2 / (np.pi * a)) + (ln_part / 2)
    term2 = ln_part / a

    return np.sign(x) * np.sqrt(np.sqrt(term1**2 - term2) - term1)

class CopulaBasedSimilarity:
    def __init__(self, patch_size=8):
        self.patch_size = patch_size

    def _extract_local_features(self, image):
        """Extracts local features from a multi-channel image for copula computation."""
        # patching
        patches = []
        for i in range(0, image.shape[0], self.patch_size):
            for j in range(0, image.shape[1], self.patch_size):
                patch = image[i:i+self.patch_size, j:j+self.patch_size, :]
                if patch.shape[0] == self.patch_size and patch.shape[1] == self.patch_size:
                    patches.append(patch)
        return patches

    def _compute_ranks(self, features):
        """Compute ranks for each pixel vector as a whole in the RGB space."""
        # Compute ranks using argsort in a single pass
        return np.mean(np.argsort(np.argsort(features, axis=0), axis=0) + 1, axis=1)

    def _ppf_normal(self, quantile):
        """Compute the percent point function (inverse of CDF) for a normal distribution."""
        return math.sqrt(2) * erfinv(2 * quantile - 1)

    def _compute_copula(self, features):
        """Compute copula from image features for multi-channel images."""
        joint_ranks = self._compute_ranks(features) / (len(features) + 1) # Normalize ranks to (0,1)
        
        # Apply PPF for normal distribution to each joint rank using vectorized operations
        copula = self._ppf_normal(joint_ranks)

        return np.nan_to_num(copula, nan=0.0, posinf=0.0, neginf=0.0) # flattened copula

    def _euclidean_distance(self, vec1, vec2):
        """Compute Euclidean distance manually."""
        return np.linalg.norm(vec1 - vec2) 

    def compute_local_similarity(self, image1, image2):
        """Compute the copula-based similarity between two images locally."""
        patches1 = self._extract_local_features(image1)
        patches2 = self._extract_local_features(image2)

        local_similarities = []
        for patch1, patch2 in zip(patches1, patches2):
            features1 = patch1.reshape(-1, patch1.shape[2])
            features2 = patch2.reshape(-1, patch2.shape[2])

            copula1 = self._compute_copula(features1)
            copula2 = self._compute_copula(features2)

            # Compute similarity metric using optimized distance calculation
            euc_distance = self._euclidean_distance(copula1, copula2) / np.sqrt(len(copula1))
            similarity = max(0, 1 - euc_distance)  # Clamp to [0, 1]
            local_similarities.append(similarity)

        return np.array(local_similarities).reshape((image1.shape[0] // self.patch_size, image1.shape[1] // self.patch_size))
