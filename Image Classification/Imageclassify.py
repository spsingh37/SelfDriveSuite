import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Transformation for training set
# See https://pytorch.org/vision/stable/transforms.html
def transform_train():
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32, 32), antialias=True),
     transforms.RandomHorizontalFlip(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  return transform

# Transformation for test set
def transform_test():
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32, 32), antialias=True),
     transforms.RandomHorizontalFlip(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  return transform


# Car dataset
class Cars(Dataset):
  def __init__(self, root, transform=None, device="cpu"):
    self.root = root # Location of files
    self.image_labels = np.genfromtxt(os.path.join(self.root, 'Labels.txt'))
    self.image_files = []
    for i in range(self.image_labels.shape[0]):
      image_file = os.path.join(self.root, str(i)+".png")
      self.image_files.append(image_file)
    self.transform=transform # Your transform
    self.device = device
  
  # Returns the length of an epoch
  def __len__(self):
    return len(self.image_files)

  # Get an individual training item by index
  def __getitem__(self, index):
    # Obtain the file name corresponding to index
    file_name = self.image_files[index]
    # Open the image
    img = Image.open(file_name)
    # Convert the image to a numpy array
    img_mat = np.copy(np.asarray(img)[:, :, :3])
    # Obtain the image classification ground truth
    file_label = int(self.image_labels[index])
    return self.transform(img_mat), file_label

  # Combine data examples into a batch
  def collate_fn(self, data):
    B = len(data)
    img_batch = torch.stack([data[i][0] for i in range(B)])
    label_batch = torch.tensor([data[i][1] for i in range(B)])
    return img_batch, label_batch






# class ImNet(nn.Module):
#   def __init__(self):
#     super().__init__()
#     # self.net = None
#     self.conv1 = nn.Conv2d(3, 64, 5)
#     self.batchnorm1 = nn.BatchNorm2d(64)
#     self.pool = nn.MaxPool2d(2, 2)
#     self.conv2 = nn.Conv2d(64, 32, 5)
#     self.batchnorm2 = nn.BatchNorm2d(32)
#     self.fc1 = nn.Linear(32 * 5 * 5, 120)
#     self.fc2 = nn.Linear(120, 84)
#     self.fc3 = nn.Linear(84, 3)

#   def forward(self, x):
#     # x = self.net(x)
#     x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
#     x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
#     x = torch.flatten(x, 1) 
#     x = F.relu(self.fc1(x))
#     x = F.relu(self.fc2(x))
#     x = self.fc3(x)
#     return x

class ImNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)  # Output: (64, 28, 28)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)  # Output: (64, 14, 14)
        self.conv2 = nn.Conv2d(64, 32, 5)  # Output: (32, 10, 10)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)  # Output: (32, 5, 5)
        self.conv3 = nn.Conv2d(32, 16, 3)  # Output: (16, 3, 3)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 3 * 3, 120)  # Flattened dimension = 16 * 3 * 3
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))  # Output: (64, 14, 14)
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))  # Output: (32, 5, 5)
        x = F.relu(self.batchnorm3(self.conv3(x)))  # Output: (16, 3, 3)
        x = torch.flatten(x, 1)  # Flatten to (batch_size, 16*3*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class ImNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 64, 5)  # Output: (64, 60, 60)
#         self.batchnorm1 = nn.BatchNorm2d(64)
#         self.pool = nn.MaxPool2d(2, 2)  # Output: (64, 30, 30)
#         self.conv2 = nn.Conv2d(64, 32, 5)  # Output: (32, 26, 26)
#         self.batchnorm2 = nn.BatchNorm2d(32)
#         self.pool2 = nn.MaxPool2d(2, 2)  # Output: (32, 13, 13)
#         self.conv3 = nn.Conv2d(32, 16, 3)  # Output: (16, 11, 11)
#         self.batchnorm3 = nn.BatchNorm2d(16)
#         self.fc1 = nn.Linear(16 * 11 * 11, 120)  # Flattened dimension = 16 * 11 * 11
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 3)

#     def forward(self, x):
#         x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))  # Output: (64, 30, 30)
#         x = self.pool2(F.relu(self.batchnorm2(self.conv2(x))))  # Output: (32, 13, 13)
#         x = F.relu(self.batchnorm3(self.conv3(x)))  # Output: (16, 11, 11)
#         x = torch.flatten(x, 1)  # Flatten to (batch_size, 16 * 11 * 11)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x