import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
import os
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
from fn_utils import HAM10kCustom, train_step, test_step, train, create_sampled_dataloader
from ResNet import res_net_101, res_net_101, res_net_101


################################################################################################

df = pd.read_csv("skin_cancer_df.csv")
X, y = df["image_path"], df["dx_expanded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)


################################################################################################

# Transformers and Data Augmentation
train_transforms = v2.Compose([
    v2.Resize(size=(96, 96)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=(-30, 30)),
    v2.TrivialAugmentWide(num_magnitude_bins=35),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.15),
    v2.ToImage(),
    v2.ToDtype(torch.float16, scale=True),
    v2.RandomErasing(p=0.1),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = v2.Compose([
    v2.Resize(size=(96, 96)),
    v2.ToImage(),
    v2.ToDtype(torch.float16, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

################################################################################################

# LR scheduler, Weight Decay and Label Smoothing

# lr_scheduler = 'cosineannealinglr'
lr_warm_epochs = 5
lr_warm_method = 'linear'
lr_warm_decay = 0.001
label_smoothing = 0.05
weight_decay = 1e-04  # Original suggested value 2e-05
long_training_epochs = 300  # Pytorch suggests 600 epochs, but that would require > 6 hours of training

################################################################################################

train_data = HAM10kCustom(X_train, y_train, train_transforms)
val_data = HAM10kCustom(X_val, y_val, test_transforms)
test_data = HAM10kCustom(X_test, y_test, test_transforms)

################################################################################################


def main():

    # Initializing model and DataLoader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net_101 = res_net_101(img_channels=3,
                          num_classes=7,
                          expansion_factor=4,
                          block_input_layout=(32, 64, 128, 256)).to(device)


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

    # Create samples of 3000, 500, 500 for ResNet101, and ResNet152
    sampled_train_dataloader = create_sampled_dataloader(original_dataloader=train_custom_dataloader,
                                                         num_samples=5000,
                                                         batch_size=BATCH_SIZE,
                                                         num_workers=NUM_WORKERS,
                                                         shuffle=True)

    sampled_val_dataloader = create_sampled_dataloader(original_dataloader=val_custom_dataloader,
                                                       num_samples=1000,
                                                       batch_size=BATCH_SIZE,
                                                       num_workers=NUM_WORKERS,
                                                       shuffle=False)

    sampled_test_dataloader = create_sampled_dataloader(original_dataloader=test_custom_dataloader,
                                                        num_samples=1000,
                                                        batch_size=BATCH_SIZE,
                                                        num_workers=NUM_WORKERS,
                                                        shuffle=False)

    ##############################################################################################

    # Initialize optimizer, loss function and LR Warmup
    torch.manual_seed(25)
    torch.cuda.manual_seed(25)

    lr = 0.01
    num_epochs = 350
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optim = torch.optim.Adam(params=net_101.parameters(),
                             weight_decay=weight_decay,
                             lr=lr)

    ##############################################################################################

    # Start training
    
    training_results = train(model=net_101,
                             train_dataloader=sampled_train_dataloader,
                             test_dataloader=sampled_val_dataloader,
                             optimizer=optim,
                             loss_function=loss_fn,
                             epochs=num_epochs,
                             device=device,
                             min_lr=1e-7,
                             patience=15,
                             cooldown=5,
                             lr_warmup_epochs=3,
                             lr_warmup_decay=lr_warm_decay,
                             iter_max=25,
                             dynamic_iter_max=True,
                             num_classes=7)

    test_loss, test_acc, test_recall, test_roc_auc = test_step(model=net_101,
                                                               dataloader=sampled_test_dataloader,
                                                               loss_function=loss_fn,
                                                               num_classes=7,
                                                               device=device)

    ##############################################################################################
    # Save state dict
    model_save_dir = os.path.join(os.getcwd(), "models")
    net_101.name = "ResNet101_v5"
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    torch.save(net_101.state_dict(), os.path.join(model_save_dir, "net_101_v5.pth"))

    ##############################################################################################
    # Write results

    train_loss, train_acc = training_results["train_loss"][-1], training_results["train_acc"][-1].item()
    train_recall = training_results["train_recall"][-1].item()
    train_roc_auc = training_results["train_roc_auc"][-1].item()
    val_loss, val_acc = training_results["test_loss"][-1], training_results["test_acc"][-1].item()
    val_recall = training_results["test_recall"][-1].item()
    val_roc_auc = training_results["test_roc_auc"][-1].item()

    with open(f"{net_101.name}.csv", "w") as f:
        f.write("Train loss,Train accuracy,Train recall,Train ROC-AUC, "
                "Validation loss,Validation Accuracy,Validation Recall,Validation ROC-AUC"
                "Test loss,Test Accuracy,Test Recall,Test ROC-AUC\n")
        f.write(f"{train_loss},{train_acc},{train_recall}{train_roc_auc},"
                f"{val_loss},{val_acc},{val_recall},{val_roc_auc}"
                f"{test_loss},{test_acc},{test_recall},{test_roc_auc}")

    with open("epoch_lrs.csv", "w") as f:
        f.write("Epoch,Lr,Trloss,TrAcc,TrRecall,TrROC,ValLoss,ValAcc,ValRecall,ValRoc\n")
        for epoch in range(num_epochs):
            line = f"{epoch+1},"
            for key in training_results.keys():
                line += f"{training_results[key][epoch]},"
            line += f"\n".replace(",\n","\n")
            f.write(line)

    print(f"Train loss: {train_loss:.3f} | ACC: {train_acc*100:.2f}% | "
          f" Recall: {train_recall*100:.2f} | AUROC: {train_roc_auc*100:.2f}")
    print(f"Validation loss: {val_loss:.3f} | ACC: {val_acc * 100:.2f}% | "
          f" Recall: {val_recall*100:.2f} | AUROC: {val_roc_auc*100:.2f}")
    print(f"Test loss: {test_loss:.3f} | ACC: {test_acc * 100:.2f}% | "
          f" Recall: {test_recall*100:.2f} | AUROC: {test_roc_auc*100:.2f}")


if __name__ == "__main__":
    main()
