import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchmetrics import Accuracy
import numpy as np
import os
from sklearn.model_selection import train_test_split
import seaborn as sns
import plotly.express as px
import warnings
from PIL import Image
from torchvision.transforms import v2
import random
from torchinfo import summary
from typing import Tuple
from tqdm.auto import tqdm
from fn_utils import HAM10kCustom
from ResNet import res_net_50, res_net_101, res_net_152

################################################################################################

df = pd.read_csv("skin_cancer_df.csv")
X, y = df["image_path"], df["dx_expanded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

labels = list(np.unique(y))
labels_to_idx = {label:idx for label, idx in zip(labels, list(range(len(labels))))}

################################################################################################

# Transformers and Data Augmentation
train_transforms = v2.Compose([
    v2.Resize(size=(224, 224)),
    v2.TrivialAugmentWide(num_magnitude_bins=45),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomErasing(p=0.1)
])

test_transforms = v2.Compose([
    v2.Resize(size=(224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

################################################################################################

# LR scheduler, Weight Decay and Label Smoothing
lr = 0.5,
lr_scheduler = 'cosineannealinglr'
lr_warmup_epochs = 5
lr_warmup_method = 'linear'
lr_warmup_decay = 0.01
label_smoothing = 0.1
weight_decay = 2e-05
long_training_epochs = 600

################################################################################################

train_data = HAM10kCustom(X_train, y_train, train_transforms)
val_data = HAM10kCustom(X_val, y_val, test_transforms)
test_data = HAM10kCustom(X_test, y_test, test_transforms)
device = "cuda" if torch.cuda.is_available() else "cpu"

################################################################################################


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device_: str = "cpu",
               compute_accuracy: bool = True):
    model.train()
    train_loss, train_acc = 0, 0 if compute_accuracy else None
    for batch, (X_, y_) in enumerate(dataloader):
        X_, y_ = X_.to(device_), y_.to(device_)
        y_pred = model(X_)

        loss = loss_function(y_pred, y_)
        train_loss += loss.item()

        if compute_accuracy:
            accuracy_fn = Accuracy(task="multiclass",
                                   num_classes=len(train_data.classes)).to(device_)
            train_acc += accuracy_fn(y_pred, y_)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_function: torch.nn.Module,
              compute_accuracy: bool = True,
              device_: str = "cpu"):

    model.eval()
    test_loss, test_acc = 0, 0 if compute_accuracy else None
    with torch.inference_mode():
        for batch, (X_, y_) in enumerate(dataloader):
            X_, y_ = X_.to(device_), y_.to(device_)

            test_preds = model(X_)

            loss = loss_function(test_preds, y_)
            test_loss += loss.item()

            if compute_accuracy:
                accuracy_fn = Accuracy(task="multiclass",
                                   num_classes=len(train_data.classes)).to(device_)
                test_acc += accuracy_fn(test_preds, y_)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    return test_loss, test_acc


def train(model:torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_function: torch.nn.Module=nn.CrossEntropyLoss(),
          epochs: int = 5,
          device_: "str" = 'cpu'):

    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_function=loss_function,
                                           optimizer=optimizer,
                                           device_=device_,
                                           compute_accuracy=True)

        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_function=loss_function,
                                        device_=device_,
                                        compute_accuracy=True)

        print(f"Epoch: {epoch}\n{'-'*20}")
        print(f"Train loss: {train_loss:.3f} | Train acc: {train_acc*100:.2f}%")
        print(f"Test loss: {test_loss:.3f} | Test acc: {test_acc*100:.2f}%")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
