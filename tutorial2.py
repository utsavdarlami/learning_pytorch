import torch
import numpy as np


print(torch.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device} is available")

# Initializing tensors in pytorch

# creating identity matrix of 5,5
x = torch.eye(5, 5)
# print(x)

# Creating a diagonal matrix with the following values in the diagonal
ele = torch.tensor([1, 2, 3])
x = torch.diag(ele)
# print(x)

# create a normal distribution tensor

x = torch.empty(size=(5, 5)).normal_(mean=8, std=1)
# print(x)

# how to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
print(tensor)
# print(tensor.bool())  # boolean / True or False
# print(tensor.short())  # int16
# print(tensor.long())  # int64 (important)
# print(tensor.half())  # float16
# print(tensor.float())  # float32 (important)
# print(tensor.double())  # float64
# print(tensor.get_device())

# Nunpy Array to tenosr and vice versa

np_array = np.zeros((3, 3))
print(np_array)

tensor_np = torch.from_numpy(np_array)
print(tensor_np)

np_array_back = tensor_np.numpy()
print(np_array_back)
