import torch
from transformers import SamModel

def load_model(model_name):
    model = SamModel.from_pretrained(model_name)
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    return model

def save_checkpoint(model, optimizer, epoch, losses, val_losses, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'val_losses': val_losses
    }
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    if torch.cuda.is_available() and checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        val_losses = checkpoint['val_losses']
        return epoch, losses, val_losses
    return 0, [], []
