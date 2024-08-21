from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchmetrics import Accuracy, Recall, AUROC
from tqdm.auto import tqdm
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, OneCycleLR, ReduceLROnPlateau
from typing import Literal


class HAM10kCustom(Dataset):
    def __init__(self, images, image_labels, transform=None):
        self.X = images.reset_index(drop=True)
        self.y = image_labels.reset_index(drop=True)
        self.classes = np.unique(self.y)
        self.transform = transform
        self.classes_to_idx = {label: label_idx for label_idx, label in enumerate(self.classes)}

    def load_image(self, index):
        """Opens an image via a path and returns it."""
        image_path = self.X[index]
        return Image.open(image_path)

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.X)

    def __getitem__(self, index):
        """Retrieves an image and its corresponding class index from the dataset."""
        img = self.load_image(index)
        class_name = self.y[index]
        class_idx = self.classes_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx


def display_random_images(dataset, classes, n=5, display_shape=False, seed=None):
    if n > 5:
        n = 5
        display_shape = False
        print(f"For display purposes n shouldn't be greater than 5. Setting n to 5")
        print(f"For values greater than 5 call the function n//5 times adjusting n subsequently.")
    if seed:
        random.seed(seed)

    random_samples_idx = random.sample(range(len(dataset)), k=n)
    plt.figure(figsize=(16, 8))
    for i, target_sample in enumerate(random_samples_idx):
        target_image, target_label = dataset[target_sample][0], dataset[target_sample][1]
        # target_image_transform = v2.Compose([
        #     v2.ToImage(),
        #     v2.ToDtype(torch.float32, scale=True)
        # ])
        # target_image = target_image_transform(target_image)
        target_image_adjust = target_image.permute(1, 2, 0)
        plt.subplot(1, n, i + 1)
        plt.imshow(target_image_adjust)
        plt.axis("off")
        if classes:
            title = f"Class: {classes[target_label]}"
            if display_shape:
                title += f"\nshape: {target_image_adjust.shape}"
            plt.title(title)
    plt.show()
    

def create_sampled_dataloader(original_dataloader, num_samples, batch_size, num_workers, shuffle=False):

    total_samples = len(original_dataloader.dataset)
    sampled_indices = random.sample(range(total_samples), num_samples)
    sampled_data = Subset(original_dataloader.dataset, sampled_indices)
    sampled_dataloader = DataLoader(dataset=sampled_data,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    shuffle=shuffle)

    return sampled_dataloader


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scaler: torch.amp.GradScaler,
               device: str = "cpu",
               num_classes: int = 2,
               compute_metrics: bool = True):
    model.to(device)
    model.train()
    train_loss, train_acc = 0, 0 if compute_metrics else None
    train_recall, train_roc_auc = 0, 0 if compute_metrics else None
    for batch, (X_, y_) in enumerate(dataloader):
        X_, y_ = X_.to(device), y_.to(device)

        with torch.autocast(device):
            y_pred = model(X_).to(device)
            loss = loss_function(y_pred, y_)

        train_loss += loss.item()

        if compute_metrics:
            accuracy_fn = Accuracy(task="multiclass",
                                   num_classes=num_classes).to(device)
            recall_fn = Recall(task="multiclass",
                               num_classes=num_classes,
                               average="weighted").to(device)
            roc_auc_fn = AUROC(task="multiclass",
                               num_classes=num_classes).to(device)
            train_acc += accuracy_fn(y_pred, y_)
            train_recall += recall_fn(y_pred, y_)
            train_roc_auc += roc_auc_fn(y_pred, y_)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    train_recall /= len(dataloader)
    train_roc_auc /= len(dataloader)
    return train_loss, train_acc, train_recall, train_roc_auc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_function: torch.nn.Module,
              scaler: amp.GradScaler=None,
              num_classes: int = 2,
              compute_metrics: bool = True,
              device: str = "cpu"):

    model.to(device)
    model.eval()
    test_loss, test_acc = 0, 0 if compute_metrics else None
    test_recall, test_roc_auc = 0, 0 if compute_metrics else None
    with torch.inference_mode():
        for batch, (X_, y_) in enumerate(dataloader):
            X_, y_ = X_.to(device), y_.to(device)

            with torch.autocast(device):
                test_preds = model(X_).to(device)
                loss = loss_function(test_preds, y_)

            test_loss += loss.item()

            if compute_metrics:
                accuracy_fn = Accuracy(task="multiclass",
                                       num_classes=num_classes).to(device)
                recall_fn = Recall(task="multiclass",
                                   num_classes=num_classes,
                                   average="weighted").to(device)
                roc_auc_fn = AUROC(task="multiclass",
                                   num_classes=num_classes).to(device)
                test_acc += accuracy_fn(test_preds, y_)
                test_recall += recall_fn(test_preds, y_)
                test_roc_auc += roc_auc_fn(test_preds, y_)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        test_recall /= len(dataloader)
        test_roc_auc /= len(dataloader)

    return test_loss, test_acc, test_recall, test_roc_auc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_function: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device: "str" = 'cpu',
          num_classes: int = 2,
          lr_warmup_epochs: int = 5,
          mode: Literal["min", "max"] = 'min',
          min_lr: float = 1e-5,
          patience: int = 5,
          cooldown: int = 5,
          lr_warmup_decay: float = 0.01,
          iter_max: int = 50,
          dynamic_iter_max: bool = False,
          max_lr=0.5,
          save_lrs: bool = True):

    scaler = torch.amp.GradScaler()

    if dynamic_iter_max:
        iter_max = len(train_dataloader) * epochs

    def linear_warmup(epoch):
        if epoch < lr_warmup_epochs:
            return lr_warmup_decay + (1 - lr_warmup_decay) * epoch / lr_warmup_epochs
        return 1.0

    results = {"lrs": [],
               "train_loss": [],
               "train_acc": [],
               "train_recall": [],
               "train_roc_auc": [],
               "test_loss": [],
               "test_acc": [],
               "test_recall": [],
               "test_roc_auc": []}

    lr_scheduler_warmup = LambdaLR(optimizer, lr_lambda=linear_warmup)
    # cosine_scheduler = CosineAnnealingLR(optimizer, T_max=iter_max - lr_warmup_epochs)
    # one_cycle_scheduler = OneCycleLR(optimizer,
    #                                  max_lr=max_lr,
    #                                  steps_per_epoch=len(train_dataloader),
    #                                  epochs=epochs,
    #                                  pct_start=lr_warmup_epochs / epochs,
    #                                  anneal_strategy='linear',  # You can use 'linear' if you prefer a linear schedule
    #                                  div_factor=25,
    #                                  final_div_factor=5000,
    #                                  three_phase=False)

    lr_on_plateau_scheduler = ReduceLROnPlateau(optimizer,
                                                mode=mode,
                                                patience=patience,
                                                cooldown=cooldown)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_recall, train_roc_auc = train_step(model=model,
                                                                        dataloader=train_dataloader,
                                                                        loss_function=loss_function,
                                                                        optimizer=optimizer,
                                                                        scaler=scaler,
                                                                        device=device,
                                                                        num_classes=num_classes,
                                                                        compute_metrics=True)

        test_loss, test_acc, test_recall, test_roc_auc = test_step(model=model,
                                                                   dataloader=test_dataloader,
                                                                   loss_function=loss_function,
                                                                   device=device,
                                                                   num_classes=num_classes,
                                                                   scaler=scaler,
                                                                   compute_metrics=True)

        if epoch < lr_warmup_epochs:
            lr_scheduler_warmup.step()
        else:
            lr_on_plateau_scheduler.step(metrics=train_loss)

        print(f"\nEpoch: {epoch+1} | Current lr: {optimizer.param_groups[0]['lr']}\n{'-'*20}")
        print(f"Train loss: {train_loss:.3f} | Acc: {train_acc*100:.2f}% |"
              f" Recall: {train_recall*100:.2f}% | AUROC: {train_roc_auc*100:.2f}%")
        print(f"Test loss: {test_loss:.3f} | Acc: {test_acc*100:.2f}%"
              f" Recall: {test_recall*100:.2f}% | AUROC: {test_roc_auc*100:.2f}%")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_recall"].append(train_recall)
        results["train_roc_auc"].append(train_roc_auc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_recall"].append(test_recall)
        results["test_roc_auc"].append(test_roc_auc)

        if save_lrs:
            results["lrs"].append(optimizer.param_groups[0]['lr'])

    return results
