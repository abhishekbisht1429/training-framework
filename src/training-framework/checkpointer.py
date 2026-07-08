import os
import torch

def save_checkpoint(checkpoint, save_dir="checkpoints", prefix="model"):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # File path
    filepath = os.path.join(save_dir, f"{prefix}_{checkpoint['timestamp']}.pt")

    # Save
    torch.save(checkpoint, filepath)

    return filepath

def load_checkpoint(filepath, map_location="cpu"):
    checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)

    return checkpoint