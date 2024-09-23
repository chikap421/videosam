from setuptools import setup, find_packages

setup(
    name="VideoSAM",
    version="0.1.0",
    author="Chika Maduabuchi",
    author_email="chika691@mit.edu",
    description="VideoSAM is a project for training and inference using SAM model for high-speed video segmentation.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/chikap421/videosam",
    packages=find_packages(),
    install_requires=[
        "datasets",
        "monai",
        "patchify",
        "tifffile",
        "matplotlib",
        "scipy",
        "seaborn",
        "scikit-learn",
        "peft",
        "numpy",
        "torch",
        "h5py",
        "tqdm",
        "optuna",
        "pillow",
        "git+https://github.com/facebookresearch/segment-anything.git",
        "git+https://github.com/huggingface/transformers.git"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)

