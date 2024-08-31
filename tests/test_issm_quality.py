import cv2
from similarity_metrics.issm_quality import ISSMsimilarity

def test_issm_similarity():
    issm = ISSMsimilarity()

    # Load images
    image1 = cv2.imread('examples/1600.png')
    image2 = cv2.imread('examples/1600.BLUR.4.png')

    # Compute ISSM value
    value = issm.issm(image1, image2)

    # Assertions
    assert value is not None, "ISSM value should not be None"
    assert isinstance(value, float), "ISSM value should be a float"

def test_issm_identical_images():
    issm = ISSMsimilarity()

    image1 = cv2.imread('examples/1600.png')
    image2 = cv2.imread('examples/1600.png')

    value = issm.issm(image1, image2)

    assert value is not None
    assert value == 1.0, "ISSM value should be 1.0 for identical images"
