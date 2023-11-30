import torch
from torch import nn
import numpy as np 


# metrics
def quadratic_cost(x, y):
    cost = torch.cdist(x, y, p=2)
    return cost/cost.max()

def cosine_cost(x, y):
    return 1 - nn.functional.cosine_similarity(x, y) 


    
class SelfOT():
    
    def __init__(self, metric=quadratic_cost, reg=0.1, iterations=100, diag_weight=10):
        super(SelfOT).__init__()
        self.metric =metric
        self.reg = reg 
        self.iterations = iterations 
        self.diag_weight = diag_weight 


    def solver(self, cost, tol=1e-6, tau=100):
        n = cost.shape[0]
        p_1 = torch.ones(n)/n 
        p_2 = torch.ones(n)/n 
        K = p_1.reshape(-1,1) * torch.exp(-cost/self.reg) * p_2.reshape(1,-1)
        # Domain stabilizators
        alpha = torch.zeros_like(p_1)
        beta = torch.zeros_like(p_2)
        # Schrodinger potentials
        u = torch.ones_like(p_1)
        v = torch.ones_like(p_2)
        # Sinkhorn itrations
        with torch.no_grad():
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
        return n*coupling 


    def masking(self, mat):
        if mat.dim() > 2:
                mat[torch.eye(mat.shape[1]).repeat(mat.shape[0], 1, 1).bool()] = self.diag_weight
        else:
            mat.fill_diagonal_(self.diag_weight)
        return mat

    
    def __call__(self, x):
        cost = self.metric(x, x)
        masked_cost = self.masking(cost)
        embedding = self.solver(masked_cost)
        return embedding


