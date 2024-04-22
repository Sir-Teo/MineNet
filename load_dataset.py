import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import tifffile as tiff
import numpy as np
from sklearn.model_selection import train_test_split
import random

class SatelliteDataset(Dataset):
    def __init__(self, data_dir, split, bands, val_size=0.2, test_size=0.1, random_state=42, oversampling=True):
        self.data_dir = data_dir
        self.split = split
        self.bands = [int(band) for band in bands.split(",")]
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.oversampling = oversampling
        self.transform = self._get_transform()
        self.image_files, self.labels = self._load_data()

    def _load_data(self):
        image_files = []
        labels = []
        label_file = os.path.join(self.data_dir, "train/answer.csv")
        with open(label_file, "r") as f:
            for line in f:
                image_file, label = line.strip().split(",")
                image_files.append(os.path.join(self.data_dir, "train/train", image_file))
                labels.append(int(label))

        # Set random seed for reproducibility
        random.seed(self.random_state)

        # Shuffle the data
        data = list(zip(image_files, labels))
        random.shuffle(data)
        image_files, labels = zip(*data)

        # Split the data into train, validation, and test sets
        num_samples = len(image_files)
        num_val_samples = int(num_samples * self.val_size)
        num_test_samples = int(num_samples * self.test_size)

        if self.split == "train":
            if self.oversampling:
                image_files, labels = self._oversample_minority_class(image_files[:-num_val_samples-num_test_samples],
                                                                      labels[:-num_val_samples-num_test_samples])
            return image_files[:-num_val_samples-num_test_samples], labels[:-num_val_samples-num_test_samples]
        elif self.split == "val":
            return image_files[-num_val_samples-num_test_samples:-num_test_samples], labels[-num_val_samples-num_test_samples:-num_test_samples]
        elif self.split == "test":
            return image_files[-num_test_samples:], labels[-num_test_samples:]
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def _oversample_minority_class(self, image_files, labels):
        minority_class = 1 - int(sum(labels) / len(labels) >= 0.5)
        minority_indices = [i for i, label in enumerate(labels) if label == minority_class]
        majority_indices = [i for i, label in enumerate(labels) if label != minority_class]

        num_oversamples = len(majority_indices) - len(minority_indices)
        oversampled_indices = random.choices(minority_indices, k=num_oversamples)

        oversampled_image_files = [image_files[i] for i in oversampled_indices]
        oversampled_labels = [labels[i] for i in oversampled_indices]

        image_files = list(image_files) + oversampled_image_files
        labels = list(labels) + oversampled_labels

        return image_files, labels

    def _get_transform(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform

    def __getitem__(self, index):
        image_file = self.image_files[index]
        label = self.labels[index]
        image = tiff.imread(image_file)
        selected_bands = image[:, :, self.bands]  # Select the desired bands
        selected_bands = np.transpose(selected_bands, (2, 0, 1))  # Change to (C, H, W) format
        selected_bands = torch.from_numpy(selected_bands)  # Convert to PyTorch tensor
        if self.transform:
            selected_bands = self.transform(selected_bands)
        selected_bands = selected_bands.to(torch.float32)  # Convert to the desired data type
        return selected_bands, label

    def __len__(self):
        return len(self.image_files)