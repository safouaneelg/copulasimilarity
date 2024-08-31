from setuptools import setup, find_packages

setup(
    name='CopulaSimilarity', 
    version='0.1.1',  
    author='Safouane El Ghazouali',
    author_email='safouane.elghazouali@gmail.com',  
    description='A package for locally sensitive Copula-Based Similarity Metric along with image quality metrics.', 
    long_description=open('README.md').read(), 
    long_description_content_type='text/markdown',  
    url='https://github.com/safouaneelg/copulasimilarity',
    packages=find_packages(), 
    install_requires=[
        'numpy==1.24.3'
    ],
    extras_require={
        'full': [
            'opencv_contrib_python==4.8.0.76',
            'opencv_python==4.8.1.78',
            'opencv_python_headless==4.7.0.72',
            'phasepack==1.5',
            'scipy==1.9.1',
            'skimage==0.0',
            'image_similarity_measures',
            'matplotlib'
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8', 
)
