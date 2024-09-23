import numpy as np
import torch

def calculate_metrics(pred_masks, true_masks):
    pred_masks = (pred_masks > 0.5).float()
    intersection = (pred_masks * true_masks).sum()
    union = (pred_masks + true_masks).sum() - intersection
    iou = intersection / union
    precision = intersection / pred_masks.sum()
    recall = intersection / true_masks.sum()
    return iou.item(), precision.item(), recall.item()

def get_grad_norms(parameters):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def adjust_grad_clip(grad_norms, grad_clip):
    if grad_norms > 5.0:
        grad_clip = max(0.1, grad_clip * 0.9)
    elif grad_norms < 0.5:
        grad_clip = min(5.0, grad_clip * 1.1)
    return grad_clip
