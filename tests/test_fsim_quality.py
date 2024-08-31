import cv2
from similarity_metrics.fsim_quality import FSIMsimilarity

def test_fsim_similarity():
    fsim = FSIMsimilarity()

    # Load images
    image1 = cv2.imread('examples/1600.png')
    image2 = cv2.imread('examples/1600.BLUR.4.png')

    # Compute FSIM value
    value = fsim.fsim(image1, image2)

    # Assertions
    assert value is not None, "FSIM value should not be None"
    assert isinstance(value, float), "FSIM value should be a float"

def test_fsim_identical_images():
    fsim = FSIMsimilarity()

    image1 = cv2.imread('examples/1600.png')
    image2 = cv2.imread('examples/1600.png')

    value = fsim.fsim(image1, image2)

    assert value is not None
    assert value == 1.0, "FSIM value should be 1.0 for identical images"
