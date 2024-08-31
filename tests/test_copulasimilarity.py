import numpy as np
import cv2
from CopulaSimilarity.CSM import CopulaBasedSimilarity

def test_copula_based_similarity():
    # Initialize the CopulaBasedSimilarity object
    csm = CopulaBasedSimilarity()

    # Load real images from the examples folder
    image1 = cv2.imread('examples/1600.png')
    image2 = cv2.imread('examples/1600.BLUR.4.png')

    # Compute local similarity
    similarity = csm.compute_local_similarity(image1, image2)

    # Assertions
    assert similarity is not None, "The similarity map should not be None"
    assert isinstance(similarity, np.ndarray), "The similarity map should be a numpy array"
    assert similarity.shape == (image1.shape[0] // csm.patch_size, image1.shape[1] // csm.patch_size), "The shape of the similarity map is incorrect"

def test_identical_images():
    # Test with completely identical images
    image1 = cv2.imread('examples/1600.png')
    image2 = cv2.imread('examples/1600.png')

    csm = CopulaBasedSimilarity()
    similarity = csm.compute_local_similarity(image1, image2)

    assert similarity is not None
    assert np.allclose(similarity, np.ones_like(similarity)), "Similarity should be 1 for identical images"

def test_different_images():
    # Test with different images
    image1 = cv2.imread('examples/1600.png')
    image2 = cv2.imread('examples/1600.jpeg2000.4.png')

    csm = CopulaBasedSimilarity()
    similarity = csm.compute_local_similarity(image1, image2)

    assert similarity is not None
    assert isinstance(similarity, np.ndarray), "The similarity map should be a numpy array"
    assert similarity.shape == (image1.shape[0] // csm.patch_size, image1.shape[1] // csm.patch_size), "The shape of the similarity map is incorrect"
