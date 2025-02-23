import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.functional import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

