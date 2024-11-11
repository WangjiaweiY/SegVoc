import torch

x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]])
print(x)
print(x.argmax(dim = 0))
print(x.argmax(dim = 1))
print(x.argmax(dim = 2))