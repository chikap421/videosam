import os
import numpy as np
import tifffile
from patchify import patchify
from datasets import Dataset as HFDataset
from PIL import Image
import random
import matplotlib.pyplot as plt

def load_tiff_stack(image_path, mask_path):
    images = tifffile.imread(image_path)
    masks = tifffile.imread(mask_path)
    return images, masks

def visualize_random_images(images, masks, num_samples=2, random_seed=42):
    random.seed(random_seed)
    random_indices = random.sample(range(images.shape[0]), num_samples)
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 8))
    for idx, ax in zip(random_indices, axes):
        ax[0].imshow(images[idx], cmap='gray')
        ax[0].set_title(f'Image {idx}')
        ax[0].axis('off')
        ax[1].imshow(masks[idx], cmap='gray')
        ax[1].set_title(f'Mask {idx}')
        ax[1].axis('off')
    plt.tight_layout()
    plt.show()
    return random_indices[0]

def patchify_images_and_masks(images, masks, patch_size, step):
    image_patches = []
    mask_patches = []
    for img, mask in zip(images, masks):
        img_patches = patchify(img, (patch_size, patch_size), step=step)
        mask_patches_ = patchify(mask, (patch_size, patch_size), step=step)

        img_patches = img_patches.reshape(-1, patch_size, patch_size)
        mask_patches_ = mask_patches_.reshape(-1, patch_size, patch_size)

        image_patches.extend(img_patches)
        mask_patches.extend(mask_patches_)
    return np.array(image_patches), np.array(mask_patches)

def filter_non_empty_patches(images, masks):
    valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]
    return images[valid_indices], masks[valid_indices]

def resize_patches(images, masks, target_size):
    resized_images = [np.array(Image.fromarray(img).resize(target_size, Image.NEAREST)) for img in images]
    resized_masks = [np.array(Image.fromarray(mask).resize(target_size, Image.NEAREST)) for mask in masks]
    return np.array(resized_images), np.array(resized_masks)

def normalize_masks(masks):
    return (masks > 0).astype(np.float32)

def create_dataset(images, masks):
    dataset_dict = {
        "image": images.tolist(),
        "label": masks.tolist(),
    }
    return HFDataset.from_dict(dataset_dict)
