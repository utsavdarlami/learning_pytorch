import torch

# Tensor reshaping

x = torch.arange(9)
x_3x3 = x.view(3, 3)
print(x_3x3)

y_v = x_3x3.t()

# contiguous because the memory should be pointed properly
print(y_v.contiguous().view(9))

x_3x3_reshape = x.reshape(3, 3)
print(x_3x3_reshape)

y_v_r = x_3x3_reshape.t()
print(y_v_r)
# print(y_v_r.reshape(9))

#  concatenate
x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))

print(torch.cat((x1, x2), dim=0).shape)
print(torch.cat((x1, x2), dim=1).shape)

z = x1.view(-1)
print(z.shape)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

#  using permute
# transpose is just the special case of permute
# permute suitable for batch tranpose
z = x.permute(0, 2, 1)
print(z.shape)

# squeeze and unsequeeze
x = torch.arange(10)  # [10]

print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

z = torch.arange(10).unsqueeze(0).unsqueeze(1)  # 1*1*10
print(z.squeeze(1).shape)
