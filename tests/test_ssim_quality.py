import cv2
from similarity_metrics.ssim_rmse_psnr_metrics import ssim

def test_ssim_similarity():
    # Load images
    image1 = cv2.imread('examples/1600.png')
    image2 = cv2.imread('examples/1600.BLUR.4.png')

    # Compute SSIM value
    value = ssim(image1, image2)

    # Assertions
    assert value is not None, "SSIM value should not be None"
    assert isinstance(value, float), "SSIM value should be a float"

def test_ssim_identical_images():

    image1 = cv2.imread('examples/1600.png')
    image2 = cv2.imread('examples/1600.png')

    value = ssim(image1, image2)

    assert value is not None
    assert value == 1.0, "SSIM value should be 1.0 for identical images"
