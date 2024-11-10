import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 56 * 56, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.reshape(-1, 16 * 56 * 56)  # Reshape for the fully connected layer
        x = self.classifier(x)
        return x

class Malaria(Dataset):
  def __init__(self, image_filepaths, transform = None):
    self.image_filepaths = image_filepaths
    self.transform = transform

  def __len__(self):
    return len(self.image_filepaths)

  def __getitem__(self, index):

    image = cv2.imread(self.image_filepaths[index])

    if(self.image_filepaths[index].split('/')[3] == 'Parasitized'):
      label = 0.0
    else:
      label = 1.0


    if self.transform:
      image_pil = Image.fromarray(image)
      image = self.transform(image_pil)
      image=image.detach().numpy()
      image=np.transpose(image, (1, 2, 0))
    return image, label
IM_SIZE=224
transform = transforms.Compose([
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.RandomRotation(degrees=(0, 90)),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Assuming RGB normalization
])
def round(x):
  if(x>= 0.5):
    return 1.
  else:
    return 0.
  
