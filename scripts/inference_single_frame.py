import torch
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor
from scripts.utils import calculate_metrics
import os

def load_and_prepare_data(image_path, mask_path):
    images = tifffile.imread(image_path)
    masks = tifffile.imread(mask_path)
    return images, masks

def predict(model, inputs):
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
        probabilities = torch.sigmoid(outputs.pred_masks.squeeze(1)).cpu().numpy().squeeze()
    return (probabilities > 0.5).astype(np.uint8)

def main():
    # Paths to the test images and masks
    image_path = "test_image.tif"
    mask_path = "test_mask.tif"
    output_dir = "masks"

    os.makedirs(output_dir, exist_ok=True)
    
    images, ground_truth_masks = load_and_prepare_data(image_path, mask_path)
    
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    model.eval()

    for idx, image in enumerate(images):
        inputs = processor(images=[image], return_tensors="pt")
        predicted_mask = predict(model, inputs)

        # Save the predicted masks
        plt.imsave(f"{output_dir}/pred_mask_{idx}.png", predicted_mask, cmap='gray')
        
        iou, precision, recall = calculate_metrics(torch.from_numpy(predicted_mask), torch.from_numpy(ground_truth_masks[idx]))
        print(f"Frame {idx}: IoU = {iou:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")

if __name__ == "__main__":
    main()
