import os
import matplotlib.pyplot as plt
import tifffile
import numpy as np

def visualize_images_and_masks(image_path, mask_path, output_dir="visualization"):
    os.makedirs(output_dir, exist_ok=True)
    
    images = tifffile.imread(image_path)
    masks = tifffile.imread(mask_path)

    for idx, (image, mask) in enumerate(zip(images, masks)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title(f'Image {idx}')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f'Mask {idx}')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/visualization_{idx}.png")
        plt.close()

def main():
    image_path = "visualize_images.tif"
    mask_path = "visualize_masks.tif"
    
    visualize_images_and_masks(image_path, mask_path)

if __name__ == "__main__":
    main()
