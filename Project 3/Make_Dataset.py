import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

class BoundingBoxDataset(Dataset):
    def __init__(self, video_data, box_data, transform=None):
        """
        Dataset for bounding box prediction grouped by patient.

        Parameters:
            video_data (dict): Dictionary where keys are patient names, values are lists of video frames (H, W, 1).
            box_data (dict): Dictionary where keys are patient names, values are lists of bounding box masks (H, W).
        """
        self.patients = list(video_data.keys())
        self.video_data = video_data
        self.box_data = box_data
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        # Get the patient name
        patient_name = self.patients[idx]

        # Get video frames and bounding box masks for this patient
        video_frames = self.video_data[patient_name]
        box_mask = self.box_data[patient_name]

        # Convert to tensors and stack
        videos_tensor = []
        for frame in video_frames:
            frame_tensor = torch.tensor(frame).permute(2, 0, 1).float() # Normalize to [0, 1]
            if self.transform:
                frame_tensor = self.transform(frame_tensor)
            videos_tensor.append(frame_tensor)

        videos_tensor = torch.stack(videos_tensor)
        boxes_tensor = torch.stack([torch.tensor(box_mask).unsqueeze(0).float() for _ in range(len(video_frames))])


        return videos_tensor, boxes_tensor


class MaskDataset(Dataset):
    def __init__(self, video_data, mask_data, transform=None):
        """
        Dataset for mask prediction grouped by patient.

        Parameters:
            video_data (dict): Dictionary where keys are patient names, values are lists of video frames (H, W, 1).
            mask_data (dict): Dictionary where keys are patient names, values are lists of mask frames (H, W).
        """
        self.patients = list(video_data.keys())
        self.video_data = video_data
        self.mask_data = mask_data
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        # Get the patient name
        patient_name = self.patients[idx]

        # Get video frames and segmentation masks for this patient
        video_frames = self.video_data[patient_name]
        mask_frames = self.mask_data[patient_name]

        # Convert to tensors and stack
        videos_tensor = []
        for frame in video_frames:
            frame_tensor = torch.tensor(frame).permute(2, 0, 1).float()  # Normalize to [0, 1]
            if self.transform:
                frame_tensor = self.transform(frame_tensor)
            videos_tensor.append(frame_tensor)

        videos_tensor = torch.stack(videos_tensor)
        masks_tensor = torch.stack([torch.tensor(mask).unsqueeze(0).float() for mask in mask_frames])


        return videos_tensor, masks_tensor
