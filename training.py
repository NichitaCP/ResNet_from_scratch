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
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

################################################################################################

df = pd.read_csv("skin_cancer_df.csv")
X, y = df["image_path"], df["dx_expanded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)


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

lr = 0.5
lr_scheduler = 'cosineannealinglr'
lr_warm_epochs = 5
lr_warm_method = 'linear'
lr_warm_decay = 0.01
label_smoothing = 0.1
weight_decay = 2e-05
long_training_epochs = 600

################################################################################################

X_train, X_test, X_val = X_train.reset_index(drop=True), X_test.reset_index(drop=True), X_val.reset_index(drop=True)
y_train, y_test, y_val = y_train.reset_index(drop=True), y_test.reset_index(drop=True), y_val.reset_index(drop=True)


train_data = HAM10kCustom(X_train, y_train, train_transforms)
val_data = HAM10kCustom(X_val, y_val, test_transforms)
test_data = HAM10kCustom(X_test, y_test, test_transforms)
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
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


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_function: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device_: "str" = 'cpu',
          lr_warmup_epochs: int = 5,
          lr_warmup_decay: float = 0.01,
          iter_max: int = 100):

    def linear_warmup(epoch):
        if epoch < lr_warmup_epochs:
            return lr_warmup_decay + (1 - lr_warmup_decay) * epoch / lr_warmup_epochs
        return 1.0

    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    lr_scheduler_warmup = LambdaLR(optimizer, lr_lambda=linear_warmup)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=iter_max - lr_warmup_epochs)

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

        if epoch < lr_warmup_epochs:
            lr_scheduler_warmup.step()
        else:
            cosine_scheduler.step()

        print(f"Epoch: {epoch+1}\n{'-'*20}")
        print(f"Train loss: {train_loss:.3f} | Train acc: {train_acc*100:.2f}%")
        print(f"Test loss: {test_loss:.3f} | Test acc: {test_acc*100:.2f}%")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


################################################################################################

# Initializing model and DataLoader
net_50_lw = res_net_50(img_channels=3,
                       num_classes=7,
                       expansion_factor=2,
                       block_input_layout=(16, 32, 64, 128)).to(device)

BATCH_SIZE = 64
NUM_WORKERS = 0
train_custom_dataloader = DataLoader(dataset=train_data,
                                     batch_size=BATCH_SIZE,
                                     num_workers=NUM_WORKERS,
                                     shuffle=True)

val_custom_dataloader = DataLoader(dataset=val_data,
                                   batch_size=BATCH_SIZE,
                                   num_workers=NUM_WORKERS,
                                   shuffle=False)

test_custom_dataloader = DataLoader(dataset=test_data,
                                    batch_size=BATCH_SIZE,
                                    num_workers=NUM_WORKERS,
                                    shuffle=False)


##############################################################################################

# Initialize optimizer, loss function and LR Warmup
torch.manual_seed(25)
torch.cuda.manual_seed(25)

NUM_EPOCHS = 100
loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optim = torch.optim.SGD(params=net_50_lw.parameters(),
                        momentum=0.9,
                        weight_decay=weight_decay,
                        lr=lr)

##############################################################################################

# Start training
training_results = train(model=net_50_lw,
                         train_dataloader=train_custom_dataloader,
                         test_dataloader=val_custom_dataloader,
                         optimizer=optim,
                         loss_function=loss_fn,
                         epochs=NUM_EPOCHS,
                         device_=device,
                         lr_warmup_epochs=lr_warm_epochs,
                         lr_warmup_decay=lr_warm_decay,
                         iter_max=25)
