import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

training_data = datasets.FashionMNIST(
    root="data",# İndirilecek dosya dizin adı
    train=True,
    download=True,
    transform=ToTensor()
)
print(training_data[0][1])# Label numarası
a=training_data[8][0][0].numpy()
b=np.array(a)
plt.imshow(b)
plt.show()
test_data = datasets.FashionMNIST(
    root="data",
    train=False,#Test data olup olmadığını belirleyen parametre
    download=True,
    transform=ToTensor()
)

# print(len(test_data))
# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     print(sample_idx)
#     img, label = training_data[sample_idx]
    
#     print("id",label)
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()


print("*0*0*0*0*0*0*0*0*0 Creating a Custom Dataset for your files")

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        
        self.img_dir = img_dir
        print("fsdg",self.img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        print("PATH",img_path)
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
image=CustomImageDataset(annotations_file='/home/erdem/Desktop/Tensor_workspace/image/labels.csv',img_dir='/home/erdem/Desktop/Tensor_workspace/image/foto')
import numpy as np
import cv2 
print("TRY",image[4])
t=image[8][0][0]
p=image[8][0][0].numpy()
s=np.array(p)
#s=s.reshape(2000,900,3)
print(s.shape)
#print(image[0][0].numpy())
plt.imshow(s)
plt.show()

print("*0*0*0*0*0*0*0*0*0 Preparing your data for training with DataLoaders")

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
print(train_dataloader)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


# Display image and label.
print("asd",next(iter(train_dataloader))[1][0])#Class elde edildi
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")