import os
import torch
import numpy as np
from transformers import SamModel, SamProcessor
from scripts.utils import calculate_metrics
import tifffile
import matplotlib.pyplot as plt

def load_composite_evaluation_data(image_dir, mask_dir):
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.tif')])
    images = [tifffile.imread(image_file) for image_file in image_files]
    masks = [tifffile.imread(mask_file) for mask_file in mask_files]
    return images, masks

def evaluate_composite_frames(model, processor, images, masks):
    model.eval()
    metrics = []
    with torch.no_grad():
        for idx, (image, mask) in enumerate(zip(images, masks)):
            inputs = processor(images=[image], return_tensors="pt")
            outputs = model(**inputs, multimask_output=False)
            predicted_mask = torch.sigmoid(outputs.pred_masks.squeeze(1)).cpu().numpy().squeeze()

            predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
            iou, precision, recall = calculate_metrics(torch.from_numpy(predicted_mask), torch.from_numpy(mask))
            metrics.append((iou, precision, recall))
            plt.imsave(f"eval_composite_pred_mask_{idx}.png", predicted_mask, cmap='gray')
    
    return metrics

def main():
    image_dir = "data/test/"
    mask_dir = "data/test/"

    images, masks = load_composite_evaluation_data(image_dir, mask_dir)

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base")

    metrics = evaluate_composite_frames(model, processor, images, masks)

    for idx, (iou, precision, recall) in enumerate(metrics):
        print(f"Composite Frame {idx}: IoU = {iou:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")

if __name__ == "__main__":
    main()
