import torch
from torch import nn
import torch.nn.functional as F


# metrics
def quadratic_cost(x, y):
    cost = torch.cdist(x, y, p=2)
    return cost/cost.max()

def cosine_cost(x, y):
    return 1 - nn.functional.cosine_similarity(x, y) 


    
import torch

class SelfOT():
    """
    Self Organizing Transformer (SelfOT) that applies the Sinkhorn algorithm to find the optimal
    transport plan for a given cost matrix. 

    Parameters:
    - tol (float): Tolerance level for the convergence of the Sinkhorn algorithm.
    - tau (float): Threshold for the stabilization trick in the Sinkhorn algorithm.
    - metric (callable): Function that computes the pairwise cost matrix between features.
    - reg (float): Regularization parameter for the Sinkhorn algorithm. Default: 0.1.
    - iterations (int): Maximum number of iterations for the Sinkhorn algorithm. Default: 10.
    - diag_weight (float): Weight to be added to the diagonal of the cost matrix. Default: 100.
    """
    def __init__(self, tol, tau, metric=quadratic_cost, reg=0.1, iterations=10, diag_weight=100):
        super(SelfOT).__init__()
        self.metric = metric
        self.reg = reg
        self.iterations = iterations
        self.diag_weight = diag_weight
        self.tol = tol
        self.tau = tau

    def solver(self, cost, tol=1e-6, tau=100):
        """
        The Sinkhorn algorithm solver that computes the optimal transport plan for the given cost matrix.

        Parameters:
        - cost (Tensor): A cost matrix whose optimal transport plan is to be computed.
        - tol (float): Tolerance for convergence. Default: 1e-6.
        - tau (float): Stabilization threshold. Default: 100.

        Returns:
        - out (Tensor): The optimal transport plan with the diagonal elements set to 1.
        """
        n = cost.shape[0]
        # Initialize the marginals uniformly
        p_1 = torch.ones(n, device='cuda') / n
        p_2 = torch.ones(n, device='cuda') / n
        # Compute the initial Kernel matrix
        K = p_1.reshape(-1, 1) * torch.exp(-cost / self.reg) * p_2.reshape(1, -1)

        # Initialize stabilizers and potentials
        alpha = torch.zeros_like(p_1)
        beta = torch.zeros_like(p_2)

        # Schrodinger potentials
        u = torch.ones_like(p_1)
        v = torch.ones_like(p_2)

        # Perform Sinkhorn iterations
        for i in range(self.iterations):
            v = p_2 / torch.matmul(K.T, u)
            u = p_1 / torch.matmul(K, v)
            coupling = u.reshape(-1, 1) * K * v.reshape(1, -1)
            error = torch.max(torch.abs(torch.sum(coupling, axis=0) - p_2))
            # Break the loop if convergence is achieved
            if error < tol:
                break
            # Apply stabilization if necessary
            if torch.max(torch.abs(u)) > tau or torch.max(torch.abs(v)) > tau:
                alpha += self.reg * torch.log(u)
                beta += self.reg * torch.log(v)
                # Recompute the Kernel matrix with updated stabilizers
                K = p_1.reshape(-1, 1) * torch.exp(-(cost - alpha.reshape(-1, 1) - beta.reshape(1, -1)) / self.reg) * p_2.reshape(1, -1)
                u = torch.ones_like(p_1)
                v = torch.ones_like(p_2)
        # Scale the coupling and set the diagonal to 1
        out = n * coupling
        return out.fill_diagonal_(1)

    def masking(self, mat):
        """
        Applies a mask to the given matrix by setting its diagonal elements to the specified diagonal weight.

        Parameters:
        - mat (Tensor): The matrix to which the masking is applied.

        Returns:
        - mat (Tensor): The masked matrix with its diagonal elements set to the specified weight.
        """
        if mat.dim() > 2:
            # Apply masking for a batch of matrices
            mat[torch.eye(mat.shape[1]).repeat(mat.shape[0], 1, 1).bool()] = self.diag_weight
        else:
            # Apply masking for a single matrix
            mat.fill_diagonal_(self.diag_weight)
        return mat

    def __call__(self, x):
        """
        Transforms the input features using the SelfOT method.

        Parameters:
        - x (Tensor): The input features to be transformed.

        Returns:
        - embedding (Tensor): The transformed features after applying SelfOT.
        """
        # Compute the cost matrix using the provided metric
        cost = self.metric(x, x)
        # Mask the cost matrix by modifying its diagonal
        masked_cost = self.masking(cost)
        # Solve the optimal transport problem on the masked cost matrix
        embedding = self.solver(masked_cost, self.tol, self.tau)
        return embedding



class SelfAttention(nn.Module):
    """
    A self-attention module that computes the attention mechanism on the input feature set.
    
    Attributes:
    - query_layer (nn.Linear): A fully connected layer that transforms input features into queries.
    - key_layer (nn.Linear): A fully connected layer that transforms input features into keys.
    - value_layer (nn.Linear): A fully connected layer that transforms input features into values.
    """

    def __init__(self):
        super(SelfAttention, self).__init__()
        # Initially, the layers are set to None 
        self.query_layer = None
        self.key_layer = None
        self.value_layer = None

    def forward(self, x):
        """
        The forward pass of the self-attention mechanism that computes weighted feature sums.
        
        Parameters:
        - x (Tensor): The input feature tensor 
        
        Returns:
        - attention_probs (Tensor): The attention probability matrix used to weight the input features.
        """

        # Dynamically create the layers if not already created
        if self.query_layer is None or self.key_layer is None or self.value_layer is None:
            dim = x.size(-1)  # Last dimension size is used as the feature dimension
            # Linear layers to transform the input features into queries, keys, and values
            self.query_layer = nn.Linear(dim, dim).to(x.device)
            self.key_layer = nn.Linear(dim, dim).to(x.device)
            self.value_layer = nn.Linear(dim, dim).to(x.device)

        # Compute query, key, and value matrices by passing the input features through the corresponding layers
        queries = self.query_layer(x)  
        keys = self.key_layer(x)       
        values = self.value_layer(x)   

        # Compute attention scores by performing a batch matrix-matrix product of the query and key matrices
        scores = torch.matmul(queries, keys.transpose(-1, -2))

        # Apply softmax to the scores to obtain attention probabilities
        attention_probs = F.softmax(scores, dim=-1)

        return attention_probs
        
