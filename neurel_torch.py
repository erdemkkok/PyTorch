import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
logits=NeuralNetwork()
print(logits)
model = NeuralNetwork().to(device)
print(model)


X = torch.rand(1,28, 28, device=device)
#print(X)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)# En yüksek dönen class değerini alır
print(f"Predicted class: {pred_probab}")
print(f"Predicted class: {y_pred}")

print("-*-*-*-*-*-* MODEL LAYERS")

import numpy as np
import matplotlib.pyplot as plt
input_image = torch.rand(3,28,28) #Random renklerden resim dosyası
# print(np.array(input_image.numpy()))
# plt.imshow((np.array(input_image.numpy())).reshape(28,28,3))
# plt.show()
print(input_image.size())
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())


print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")


"""Katmanları Sequential model içerisine tekrar tanımladı
    """
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

softmax = nn.Softmax(dim=1)# dim parametresi değerler toplamının 1 olması gerektiğini belirtir.
pred_probab = softmax(logits)


print("-*-*-*-*-*-* MODEL PARAMETERS")


print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")