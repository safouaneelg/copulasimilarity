from skimage.metrics import structural_similarity
import numpy as np

def ssim(org_img: np.ndarray, pred_img: np.ndarray, max_p: int = 255) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two RGB images.

    Parameters:
    - org_img: The original RGB image.
    - pred_img: The RGB image to compare.
    - max_p: The dynamic range of the image, typically 255 for 8-bit images.

    Returns:
    - SSIM value between the two images.
    """
    return structural_similarity(org_img, pred_img, data_range=max_p, channel_axis=2)

def rmse(org_img: np.ndarray, pred_img: np.ndarray, max_p: int = 4095) -> float:
    """
    Compute Root Mean Squared Error (RMSE) between two RGB images.

    Parameters:
    - org_img: The original RGB image.
    - pred_img: The RGB image to compare.
    - max_p: The maximum pixel value, default is 4095 for 12-bit images.

    Returns:
    - RMSE value averaged across all channels.
    """
    org_img = org_img.astype(np.float32)
    pred_img = pred_img.astype(np.float32)
    
    if org_img.ndim == 2:
        org_img = np.expand_dims(org_img, axis=-1)
    if pred_img.ndim == 2:
        pred_img = np.expand_dims(pred_img, axis=-1)

    diff = org_img - pred_img
    mse_bands = np.mean(np.square(diff / max_p), axis=(0, 1))
    rmse_bands = np.sqrt(mse_bands)
    return np.mean(rmse_bands)

def psnr(org_img: np.ndarray, pred_img: np.ndarray, max_p: int = 4095) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two RGB images.

    Parameters:
    - org_img: The original RGB image.
    - pred_img: The RGB image to compare.
    - max_p: The maximum pixel value, default is 4095 for 12-bit images.

    Returns:
    - PSNR value in dB averaged across all channels.
    """
    org_img = org_img.astype(np.float32)
    pred_img = pred_img.astype(np.float32)
    
    if org_img.ndim == 2:
        org_img = np.expand_dims(org_img, axis=-1)
    if pred_img.ndim == 2:
        pred_img = np.expand_dims(pred_img, axis=-1)

    mse_bands = np.mean(np.square(org_img - pred_img), axis=(0, 1))
    mse = np.mean(mse_bands)
    return 20 * np.log10(max_p / np.sqrt(mse))
