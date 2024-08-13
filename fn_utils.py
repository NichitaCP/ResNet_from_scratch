from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchmetrics import Accuracy
from tqdm.auto import tqdm


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
