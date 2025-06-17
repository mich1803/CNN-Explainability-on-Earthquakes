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
os.environ["PYTHONHASHSEED"] = str(SEED)

def calculate_and_save_stats(data_path, metadata, stats_file):
    print("Calculating dataset statistics for normalization...")
    df = pd.read_csv(metadata)
    all_pixels = []
    for filename in df[df.split == "train"]["filename"].values:
            img_path = os.path.join(data_path, filename)
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

def spectra_stats(data_path, meta_path):
    stats_file = os.path.join(data_path, "stats.json")
    if os.path.exists(stats_file):
        mean, std = load_stats(stats_file)
        print(f"({data_path}) Mean: {mean}, Std: {std} (calculated and saved)")
    else:
        mean, std = calculate_and_save_stats(data_path, meta_path, stats_file)
        print(f"({data_path}) Mean: {mean}, Std: {std} (loaded from file)")
    
    return mean, std

class CNN2D(pl.LightningModule):
    def __init__(self, input_dim, n_classes = 2, lr = 1e-3, dropout = 0.1):
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
        x, y = batch[:2]
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

class SpectraDataset(Dataset):
    def __init__(self, data_path, dataframe_path, split, transform=None, get_metadata=False, same_amount=True, colocated = False):
        self.data_path = data_path
        self.transform = transform
        self.df = pd.read_csv(dataframe_path)
        self.file_list = self._get_file_list(split, colocated)
        if same_amount:
            self.file_list = self._take_same_amount()
        self.get_metadata = get_metadata

    def _get_file_list(self, split, colocated):
        if split == 'train':
            if not colocated:
                return self.df[self.df["split"] == split]["filename"].values
            else:
                return self.df[(self.df["split"] == split) & (self.df["associato"] == 0)]["filename"].values
        elif split == 'test':
            if not colocated:
                return self.df[self.df["split"] == split]["filename"].values
            else:
                return self.df[(self.df["split"] == split) | (self.df["associato"] != 0)]["filename"].values
    
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
        
        trace_name = img_name.removesuffix('.png')

        row = self.df[self.df["trace_name"] == trace_name].copy()
        label_df = row["label"].values[0]
        label = 1 if label_df == 'post' else 0
        
        if self.transform:

            image = self.transform(image)

        if self.get_metadata:
            return (image, label, 
                    trace_name, 
                    float(row["p_s_diff_sec"].iloc[0]), 
                    int(row["week"].iloc[0]), 
                    int(row["associato"].iloc[0]))
        else:
            return image, label
        
    def _take_same_amount(self):
        """Balances the dataset by selecting an equal number of 'pre' and 'post' samples from self.file_list."""
        # Filter self.df to only contain rows in self.file_list
        filtered_df = self.df[self.df["filename"].isin(self.file_list)]

        # Extract filenames based on labels
        pre_files = filtered_df[filtered_df["label"] == "pre"]["filename"].tolist()
        post_files = filtered_df[filtered_df["label"] == "post"]["filename"].tolist()

        # Shuffle lists to introduce randomness
        shuffle(pre_files)
        shuffle(post_files)

        # Take the same number from both classes (if possible)
        n = min(len(pre_files), len(post_files))
        if n == 0:  # Avoid empty dataset issues
            return pre_files + post_files  # Return whatever is available

        balanced_files = pre_files[:n] + post_files[:n]
        shuffle(balanced_files)  # Shuffle again to mix 'pre' and 'post'

        return balanced_files
    

class ExplainerDataset(Dataset):
    def __init__(self, data_path, dataframe_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.df = pd.read_csv(dataframe_path)
        self.file_list = [f for f in os.listdir(data_path) if f.endswith(".png")]
    
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
        
        trace_name = img_name.removesuffix('.png')

        row = self.df[self.df["trace_name"] == trace_name].copy()
        label_df = row["label"].values[0]
        label = 1 if label_df == 'post' else 0
        
        if self.transform:
            image = self.transform(image)

        return (image, label, 
                trace_name, 
                float(row["p_s_diff_sec"].iloc[0]), 
                int(row["week"].iloc[0]), 
                int(row["associato"].iloc[0]))
    
    
