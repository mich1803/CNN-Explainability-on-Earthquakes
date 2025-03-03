import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.functional import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random
from random import shuffle
import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
import json

# Set a fixed seed value
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
pl.seed_everything(SEED)

def calculate_and_save_stats(train_data_path, stats_file):
    print("Calculating dataset statistics for normalization...")
    all_pixels = []
    for root, dirs, files in os.walk(train_data_path):
        for file in tqdm(files, desc="Calculating stats", unit="image"):
            if file.endswith(".png"):
                img_path = os.path.join(root, file)
                image = Image.open(img_path).convert("RGB")
                all_pixels.append(np.array(image) / 255.0)
    all_pixels = np.concatenate([img.reshape(-1, 3) for img in all_pixels], axis=0)
    mean = all_pixels.mean(axis=0)
    std = all_pixels.std(axis=0)
    
    # Save statistics to a file
    with open(stats_file, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
    
    
    return mean, std

def load_stats(stats_file):
    with open(stats_file, "r") as f:
        stats = json.load(f)
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
    return mean, std

def spectra_stats(data_path):
    stats_file = os.path.join(data_path, "stats.json")
    if os.path.exists(stats_file):
        mean, std = load_stats(stats_file)
        print(f"({data_path}) Mean: {mean}, Std: {std} (calculated and saved)")
    else:
        mean, std = calculate_and_save_stats(data_path, stats_file)
        print(f"({data_path}) Mean: {mean}, Std: {std} (loaded from file)")
    
    return mean, std

class CNN2D(pl.LightningModule):
    def __init__(self, input_dim = (33, 188), n_classes = 2, lr = 1e-3, dropout = 0.1):
        super(CNN2D, self).__init__()
        self.lr = lr
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.BatchNorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.BatchNorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.BatchNorm3 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.flatten = nn.Flatten()

        self.dummy = torch.zeros((1, 3, *input_dim))

        self.fc1 = nn.Linear(self._get_flattened_size(self.dummy), 128)
        self.fc2 = nn.Linear(128, self.n_classes)

        self.criterion = nn.CrossEntropyLoss()

    def _get_flattened_size(self, x):
        """Pass a dummy tensor through the conv layers to compute the flattened size."""
        with torch.no_grad():
            x = self.relu(self.maxpool(self.BatchNorm1(self.conv1(x))))
            x = self.relu(self.maxpool(self.BatchNorm2(self.conv2(x))))
            x = self.relu(self.BatchNorm3(self.conv3(x)))
            x = self.flatten(x)
        return x.shape[1]  # Get flattened size
    
    def forward(self, x):
        x = self.relu(self.maxpool(self.BatchNorm1(self.conv1(x))))
        x = self.relu(self.maxpool(self.BatchNorm2(self.conv2(x))))
        x = self.relu(self.BatchNorm3(self.conv3(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        preds = logits.argmax(dim=1)

        # Accumulate predictions and labels
        if not hasattr(self, 'total_conf_matrix'):
            self.total_conf_matrix = torch.zeros(self.n_classes, self.n_classes, device=x.device)

        # Per-batch confusion matrix
        conf_matrix = confusion_matrix(preds, y, task="multiclass", num_classes=self.n_classes)
        self.total_conf_matrix += conf_matrix


    def on_test_epoch_end(self):
        # Calculate total metrics from the confusion matrix

        conf_matrix = self.total_conf_matrix.cpu().numpy()
        plt.figure(figsize=(5, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False,
                    xticklabels=["Predicted: Foreshock", "Predicted: Aftershock"], yticklabels=["Actual: Foreshock", "Actual: Aftershock"])
        
        # Title and labels
        plt.title("Confusion Matrix")
        plt.show()
        
        TN, FP, FN, TP = self.total_conf_matrix.flatten().tolist()
        total_acc = (TP + TN) / (TP + TN + FP + FN)
        precision_score = TP / (TP + FP)
        recall_score = TP / (TP + FN)
        f1 = 2 * (precision_score * recall_score) / (precision_score + recall_score)

        # Print total metrics
        print(f"Total Accuracy: {total_acc:.4f}")
        print(f"Total Precision: {precision_score:.4f}")
        print(f"Total Recall: {recall_score:.4f}")
        print(f"Total F1 Score: {f1:.4f}")

        # Return total metrics
        return {
            "accuracy": total_acc,
            "precision": precision_score,
            "recall": recall_score,
            "f1_score": f1,
            "conf_matrix": self.total_conf_matrix,
        }
            
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = self.lr)

class SpectraDataset_p(Dataset):
    def __init__(self, data_path, transform=None, get_image_name=False, same_amount=False):
        self.data_path = data_path
        self.transform = transform
        self.file_list = [f for f in os.listdir(data_path) if f.endswith(".png")]
        if same_amount:
            self.file_list = self.take_same_amount()
        self.get_image_name = get_image_name
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_name = self.file_list[index]
        img_path = os.path.join(self.data_path, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise

        if img_name.endswith("_post.png"):
            label = 1 # Aftershock
            img_name = img_name.removesuffix("_post.png")

        elif img_name.endswith("_pre.png"):
            label = 0 # Foresehock
            img_name = img_name.removesuffix("_pre.png")

        else:
            raise ValueError(f"Invalid image name: {img_name}")
        
        if self.transform:

            image = self.transform(image)

        if self.get_image_name:
            return image, label, img_name
        else:
            return image, label
        
    def take_same_amount(self):
        pre_files = [f for f in self.file_list if f.endswith("_pre.png")]
        post_files = [f for f in self.file_list if f.endswith("_post.png")]
        shuffle(pre_files)
        shuffle(post_files)
        n = min(len(pre_files), len(post_files))
        return pre_files[:n] + post_files[:n]
    
class SpectraDataset_s(Dataset):
    def __init__(self, data_path, meta_path, transform=None, get_image_name=False, same_amount=False):
        self.data_path = data_path
        self.df = pd.read_csv(meta_path)
        self.transform = transform
        self.file_list = [f for f in os.listdir(data_path) if f.endswith(".png")]
        if same_amount:
            self.file_list = self.take_same_amount()
        self.get_image_name = get_image_name
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_name = self.file_list[index]
        img_path = os.path.join(self.data_path, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise

        if img_name.endswith("_post.png"):
            label = 1 # Aftershock
            img_name = img_name.removesuffix("_post.png")

        elif img_name.endswith("_pre.png"):
            label = 0 # Foresehock
            img_name = img_name.removesuffix("_pre.png")

        else:
            raise ValueError(f"Invalid image name: {img_name}")
        
        p_s_diff = self.df[self.df["trace_name"] == img_name].p_s_diff_sec.values[0]
        img_len = image.size[0]

        # Calculate pixels per second
        pixels_per_second = img_len / 25
        
        # Fixed crop width (corresponding to 20 seconds)
        crop_width = int(20 * pixels_per_second)
        
        # Define crop start position, ensuring a fixed width
        start_pixel = max(0, int((p_s_diff) * pixels_per_second))
        end_pixel = start_pixel + crop_width

        # Crop the image (assuming width represents time)
        image = image.crop((start_pixel, 0, end_pixel, image.height))
        
        if self.transform:

            image = self.transform(image)

        if self.get_image_name:
            return image, label, img_name
        else:
            return image, label
        
    def take_same_amount(self):
        pre_files = [f for f in self.file_list if f.endswith("_pre.png")]
        post_files = [f for f in self.file_list if f.endswith("_post.png")]
        shuffle(pre_files)
        shuffle(post_files)
        n = min(len(pre_files), len(post_files))
        return pre_files[:n] + post_files[:n]
    
