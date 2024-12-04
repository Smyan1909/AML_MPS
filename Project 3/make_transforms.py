from torchvision import transforms
import torch

def get_transforms(augment=True):

    if augment:
        transform = transforms.Compose([
                        transforms.RandomAffine(
                                degrees=15,  # Small rotations
                                scale=(0.9, 1.1),  # Zoom in/out
                                shear=10  # Shearing for probe misalignment
                            ),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Simulate contrast variations
                        transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.05),  # Add Gaussian noise
                        transforms.Normalize(mean=[0.5], std=[0.5])])# Normalize to [-1, 1]
    else:
       transform =  transforms.Normalize(mean=[0.5], std=[0.5])

    return transform