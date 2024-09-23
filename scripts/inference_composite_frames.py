import os
import torch
import numpy as np
import tifffile
from transformers import SamModel, SamProcessor
from scripts.utils import calculate_metrics
import matplotlib.pyplot as plt

def load_composite_data(image_dir, mask_dir):
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.tif')])
    images = [tifffile.imread(image_file) for image_file in image_files]
    masks = [tifffile.imread(mask_file) for mask_file in mask_files]
    return images, masks

def predict_composite_frame(model, inputs):
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
        probabilities = torch.sigmoid(outputs.pred_masks.squeeze(1)).cpu().numpy().squeeze()
    return (probabilities > 0.5).astype(np.uint8)

def main():
    image_dir = "data/train/"
    mask_dir = "data/train_mask/"
    output_dir = "composite_predictions/"

    os.makedirs(output_dir, exist_ok=True)

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    model.eval()

    images, ground_truth_masks = load_composite_data(image_dir, mask_dir)

    for idx, (image, ground_truth) in enumerate(zip(images, ground_truth_masks)):
        inputs = processor(images=[image], return_tensors="pt")
        predicted_mask = predict_composite_frame(model, inputs)

        plt.imsave(f"{output_dir}/composite_pred_mask_{idx}.png", predicted_mask, cmap='gray')

        iou, precision, recall = calculate_metrics(torch.from_numpy(predicted_mask), torch.from_numpy(ground_truth))
        print(f"Frame {idx}: IoU = {iou:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")

if __name__ == "__main__":
    main()
