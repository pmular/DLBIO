import torch 
from transform import SelfOT

n = 10
d = 50

transform = SelfOT()

X = torch.randn((n, d))

embedding = transform(X)

print(embedding.shape)


