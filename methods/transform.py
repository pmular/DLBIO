import torch
from torch import nn
import numpy as np 
import torch.nn.functional as F


# metrics
def quadratic_cost(x, y):
    cost = torch.cdist(x, y, p=2)
    return cost/cost.max()

def cosine_cost(x, y):
    return 1 - nn.functional.cosine_similarity(x, y) 


    
class SelfOT():
    def __init__(self, tol, tau, metric=quadratic_cost, reg=0.1, iterations=10, diag_weight=100):
        super(SelfOT).__init__()
        self.metric =metric
        self.reg = reg 
        self.iterations = iterations 
        self.diag_weight = diag_weight 
        self.tol = tol
        self.tau = tau


    def solver(self, cost, tol=1e-6, tau=100):
        n = cost.shape[0]
        p_1 = torch.ones(n, device='cuda')/n 
        p_2 = torch.ones(n, device='cuda')/n 
        K = p_1.reshape(-1,1) * torch.exp(-cost/self.reg) * p_2.reshape(1,-1)
        # Domain stabilizators
        alpha = torch.zeros_like(p_1)
        beta = torch.zeros_like(p_2)
        # Schrodinger potentials
        u = torch.ones_like(p_1)
        v = torch.ones_like(p_2)
        # Sinkhorn itrations
        for i in range(self.iterations):
            v = p_2 / torch.matmul(K.T, u)
            u = p_1 / torch.matmul(K, v)
            coupling = u.reshape(-1, 1) * K * v.reshape(1, -1)
            error = torch.max(torch.abs(torch.sum(coupling, axis=0)-p_2))
            if error < tol:
                break
            if torch.max(torch.abs(u)) > tau or torch.max(torch.abs(v)) > tau:
                alpha += self.reg * torch.log(u)
                beta += self.reg * torch.log(v)
                K = p_1.reshape(-1,1) * torch.exp(-(cost-alpha.reshape(-1, 1)-beta.reshape(1, -1))/self.reg) * p_2.reshape(1,-1)
                u = torch.ones_like(p_1)
                v = torch.ones_like(p_2)       
        out = n*coupling
        return out.fill_diagonal_(1)  


    def masking(self, mat):
        if mat.dim() > 2:
                mat[torch.eye(mat.shape[1]).repeat(mat.shape[0], 1, 1).bool()] = self.diag_weight
        else:
            mat.fill_diagonal_(self.diag_weight)
        return mat

    
    def __call__(self, x):
        cost = self.metric(x, x)
        masked_cost = self.masking(cost)
        embedding = self.solver(masked_cost, self.tol, self.tau)
        return embedding


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.query_layer = None
        self.key_layer = None
        self.value_layer = None

    def forward(self, x):
        # Dynamically create the layers if not already created
        if self.query_layer is None or self.key_layer is None or self.value_layer is None:
            dim = x.size(-1)
            self.query_layer = nn.Linear(dim, dim).to(x.device)
            self.key_layer = nn.Linear(dim, dim).to(x.device)
            self.value_layer = nn.Linear(dim, dim).to(x.device)

        # Compute query, key, and value matrices
        queries = self.query_layer(x)
        keys = self.key_layer(x)
        values = self.value_layer(x)  # Transform inputs into 'values'

        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-1, -2)) 

        # Apply softmax to get attention weights
        attention_probs = F.softmax(scores, dim=-1)



        return attention_probs

"""
dummy_input = torch.randn(4, 10)  # batch_size=4, seq_len=10, feature_dim=20

# Initialize the SelfAttention layer and test it with the dummy input
self_attention = SelfAttention()
attended_features = self_attention(dummy_input)

# Print the output dimensions and the output for verification
print("Attended features dimensions:", attended_features.shape)"""