import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from similarity_metrics.ssim_rmse_psnr_metrics import ssim, rmse, psnr
from similarity_metrics.fsim_quality import FSIMsimilarity
from similarity_metrics.issm_quality import ISSMsimilarity
from CopulaSimilarity.CSM import CopulaBasedSimilarity as CSMSimilarity

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Compare two images using various similarity metrics.')
    parser.add_argument('--path1', required=True, help='Path to the first image')
    parser.add_argument('--path2', required=True, help='Path to the second image')
    parser.add_argument('--issm', action='store_true', help='Compute ISSM similarity')
    parser.add_argument('--fsim', action='store_true', help='Compute FSIM similarity')
    parser.add_argument('--ssim', action='store_true', help='Compute SSIM similarity')
    parser.add_argument('--rmse', action='store_true', help='Compute RMSE')
    parser.add_argument('--psnr', action='store_true', help='Compute PSNR')
    parser.add_argument('--save_csm_map', action='store_true', help='Save the Copula-Based Similarity Map')
    parser.add_argument('--patch_size', type=int, default=8, help='Patch size for Copula-Based Similarity (default: 8)') 
    
    args = parser.parse_args()
    
    image1 = cv2.imread(args.path1)
    image2 = cv2.imread(args.path2)

    if image1 is None or image2 is None:
        print("Error: One or both image paths are incorrect.")
        return
    
    # Initialize similarity objects
    fsim_similarity = FSIMsimilarity()
    issm_similarity = ISSMsimilarity()
    copula_similarity = CSMSimilarity(patch_size=args.patch_size)  

    if args.ssim:
        ssim_value = ssim(image1, image2)
        print(f"SSIM: {ssim_value:.4f}")
    
    if args.rmse:
        rmse_value = rmse(image1, image2)
        print(f"RMSE: {rmse_value:.4f}")
    
    if args.psnr:
        psnr_value = psnr(image1, image2)
        print(f"PSNR: {psnr_value:.4f} dB")
    
    if args.issm:
        issm_value = issm_similarity.issm(image1, image2)
        print(f"ISSM: {issm_value:.4f}")
    
    if args.fsim:
        fsim_value = fsim_similarity.fsim(image1, image2)
        print(f"FSIM: {fsim_value:.4f}")
    
    csm_map = copula_similarity.compute_local_similarity(image1, image2)
    csm = np.mean(csm_map)
    print(f"CSM: {csm:.4f}")
    
    if args.save_csm_map:
        plt.figure(figsize=(8, 6))
        plt.title('Copula-Based Similarity Map')
        plt.imshow(csm_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Similarity')
        plt.savefig('csm_map.png')
        print("CSM Map saved as 'csm_map.png'")
    
if __name__ == '__main__':
    main()
