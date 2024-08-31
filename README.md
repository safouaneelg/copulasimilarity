# Copula-based Similarity Metric (CSM)
A novel locally sensitive image similarity metric based on Gaussian Copula

[SHORT EXPLANATION OF COPULA BASED SIMILARITY]

Example:
Comparative study on MRI imagery while fixing the first frame and calculating similarity metrics SSIM FSIM ISSM and CSM between that frame and subsequent ones  

# Copula-based Similarity Metric (CSM)
A novel locally sensitive image similarity metric based on Gaussian Copula.


## 📖 Overview

Copula-based Similarity Metric (CSM) is a unique approach for measuring image similarity that leverages the properties of Gaussian copulas to provide a locally sensitive measure of similarity between images. Unlike traditional metrics, CSM is designed to capture both global and local image features, making it particularly effective for applications in medical imaging, remote sensing, and any domain requiring fine-grained image comparison.

## 🌟 Features

- **Locally Sensitive**: Captures detailed differences at a granular level.
- **Gaussian Copula-Based**: Utilizes statistical properties for robust similarity measurement.
- **Versatile Usage**: Suitable for various image types and applications.
- **Extensible**: Easily integrates with other image quality metrics like SSIM, FSIM, and ISSM.


## 🚀 Getting Started

### Installation

To install the CopulaSimilarity package, you can use pip:

```bash
pip install CopulaSimilarity
```

### Usage

you can import the package and estimate the similarity map as follow:

```
from CopulaSimilarity.CSM import CopulaBasedSimilarity as CSMSimilarity

copula_similarity = CSMSimilarity()

#load your images
image1 = cv2.imread('path_to_image1')
image2 = cv2.imread('path_to_image2')

#calculate the similarity map
csm_value = copula_similarity.compute_local_similarity(image, blurred_image)

# Optionally: you can show the similarity map using cv2 or matplotlib

#if you need a metric you can calculat the mean of the copula similarity map
csm_mean = np.mean(csm_value)
```

Other metrics can also be used, the implementation is based on (image-similarity-measures)[https://github.com/nekhtiari/image-similarity-measures/tree/master] package. you can either install it using `pip install image-similarity-measures` command, or you can also use our implementation.
To use other metrics such as SSIM FSIM and ISSM, it's very similar however they only return a value. Tutorial:

```
from similarity_metrics.fsim_quality import FSIMsimilarity
from similarity_metrics.issm_quality import ISSMsimilarity

fsim_similarity = FSIMsimilarity()
issm_similarity = ISSMsimilarity()

#load your images
image1 = cv2.imread('path_to_image1')
image2 = cv2.imread('path_to_image2')

ssim_value = fsim_similarity.fsim(image1, image2)
issm_value = issm_similarity.issm(image1, image2)
```

## 📚 Example Use Case

The example below shows a comparative study on MRI imagery, where the first frame is fixed, and similarity metrics (SSIM, FSIM, ISSM, and CSM) are calculated between that frame and subsequent ones. This highlights the differences captured by each metric, demonstrating the unique sensitivity and accuracy of CSM in various scenarios.

![csm_vs_other_metrics](images/analysis4.gif)

## Licence

The work can be used for research purposes.

<!--Please cite use if you use our implementation as following:-->