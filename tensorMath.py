import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)

z2 = torch.add(x, y)
z = x + y

# Subtraction
z = x - y

# division
z = torch.true_divide(x, y)

# inplace operations
t = torch.zeros(3)
t.add_(x)

# Exponentiation
z = x.pow(2)
print(z)

z = x ** 2

# Simple comparision
z = x > 0
z = x < 0

# Matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)  # 2 * 3
x3 = x1.mm(x2)

# matrix exponentiation
matrix_exp = torch.rand(5, 5)
# print(matrix_exp.matrix_power(3))

# element wise mult
z = x * y
print(z)

# dot product
z = torch.dot(x, y)
print(z)

# Batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))

out_bmm = torch.bmm(tensor1, tensor2)  # (batch, n, p)

# Example of Broadcasting

x1 = torch.randint(1, 10, (5, 5))
x2 = torch.randint(1, 10, (1, 5))
print(x1)
print(x2)
z = x1-x2
print(z)

# other useful tensor operations
values, indices = torch.max(x, dim=0)
print(f"{values} at {indices} is max")
values, indices = torch.min(x, dim=0)
print(f"{values} at {indices} is min")

z = torch.argmax(y, dim=0)
print(z)

z = torch.argmin(y, dim=0)
print(z)

x = torch.tensor([1, 12, 4, 5, 7, 18, 9, 10])
clamped_x = torch.clamp(x, max=10)
print(clamped_x)

x = torch.tensor([1, 0, 1, 0, 0], dtype=torch.bool)
z = torch.any(x)  # checks any true / or operation
print(z)
z = torch.all(x)  # checks all true / and operations
print(z)
