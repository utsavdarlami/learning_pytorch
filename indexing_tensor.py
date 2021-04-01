import torch

# Tenosr indexing

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape)

print(x[:, 0].shape)

print(x[2, 0:10])

# Fancy indexing
x = torch.linspace(0, 10)
print(x)
indices = [2, 5, 8]
print(x[indices])

x = torch.randint(1, 10, (3, 5))
print(x)
rows = torch.tensor([1, 0, 1])
cols = torch.tensor([4, 0, 2])

print(x[rows, cols])

# advanced indexing

x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[x % 2 == 0])

# useful operations

print(torch.where(x > 5, x, x*2))  # x > 5 ? x : x * 2
print(x.ndimension())
print(x.numel())
