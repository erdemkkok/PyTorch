import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

print(x_data)

np_array = np.array(data)
print(np_array)
x_np = torch.from_numpy(np_array)# Numpy array to torch tensor
print(x_np)

x_ones = torch.ones_like(x_data)# ones matrisi oluşturur
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

shape = (2,2)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

tensor = torch.ones(4, 4)
tensor[1] = 0 #1. satır 0 olur
print(tensor)


t1 = torch.cat([tensor, tensor, tensor], dim=1) #sütun olarak tensorleri birleştirir Yanyana
print(t1)

# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}") # iki tane tensör çarpımıdır.

print(tensor, "\n")
tensor.add_(5) # Heryere 5 ekler
print(tensor)

print("*****************************************")

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")


t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)

sample_idx = torch.randint(3, size=(1,)) # 1, boyutunda max değeri 3 olan random sayılarla diziyi doldur
print("samp",sample_idx)