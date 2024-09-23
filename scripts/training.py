import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts.models import load_model, save_checkpoint, load_checkpoint
from scripts.utils import calculate_metrics, adjust_grad_clip, get_grad_norms
import logging
import time
import monai

def train(model, train_loader, val_loader, optimizer, seg_loss, device, num_epochs, logger, start_epoch, losses, val_losses, checkpoint_interval, scheduler=None):
    best_val_loss = float('inf')
    grad_clip = 1.0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_losses = []
        grad_norms_list = []
        for batch_idx, batch in enumerate(tqdm(train_loader), 1):
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=batch["pixel_values"].to(device), input_boxes=batch["input_boxes"].to(device), multimask_output=False)
                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                loss = seg_loss(predicted_masks, ground_truth_masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            grad_norms = get_grad_norms(model.parameters())
            grad_clip = adjust_grad_clip(grad_norms, grad_clip)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(mean_loss)
        logger.info(f'EPOCH: {epoch + 1}/{num_epochs}, Mean Loss: {mean_loss}')

        val_loss = validate(model, val_loader, seg_loss, device)
        val_losses.append(val_loss)
        logger.info(f'EPOCH: {epoch + 1}/{num_epochs}, Val Loss: {val_loss}')

        if epoch % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, losses, val_losses, "checkpoint.pth")

        if scheduler:
            scheduler.step(val_loss)

    return losses, val_losses

def validate(model, dataloader, seg_loss, device):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(pixel_values=batch["pixel_values"].to(device), input_boxes=batch["input_boxes"].to(device), multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks)
            val_losses.append(loss.item())
    return sum(val_losses) / len(val_losses)
