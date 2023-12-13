import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from methods.transform import SelfOT
from methods.meta_template import MetaTemplate



class CustomSOT(MetaTemplate):
    """
    CustomSOT is a class for few-shot learning which applies a Self Organizing Transformer (SelfOT)
    for transforming the embedding space to improve the few-shot learning performance.

    Attributes:
    - loss_fn (nn.Module): The loss function used for training, CrossEntropyLoss in this case.
    - transform (SelfOT): Instance of SelfOT for transforming the feature space.

    Parameters:
    - backbone (nn.Module): The backbone neural network model for feature extraction.
    - n_way (int): The number of classes (ways) in the few-shot classification task.
    - n_support (int): The number of examples per class in the support set.
    - self_ot (object): An object containing the parameters for the SelfOT transformation.
    """

def __init__(self, backbone, n_way, n_support, self_ot):
    super(CustomSOT, self).__init__(backbone, n_way, n_support)
    self.loss_fn = nn.CrossEntropyLoss()
    self.transform = SelfOT(reg=self_ot.reg, diag_weight=self_ot.diag_weight, tol=self_ot.tol, tau=self_ot.tau)

def parse_feature(self, x, is_feature):
    """
    Processes the input data through the backbone network and SelfOT transformation.

    Parameters:
    - x (Tensor or list of Tensors): The input data. 
    - is_feature (bool): Flag indicating whether the input data is already in feature form.

    Returns:
    - z_all (Tensor): The transformed features ready for few-shot classification.
    """
    if isinstance(x, list):
        x = [Variable(obj.to(self.device)) for obj in x]
    else:
        x = Variable(x.to(self.device))

    if is_feature:
        z_all = x
    else:
        # Reshape input data if it's not already features
        if isinstance(x, list):
            x = [obj.contiguous().view(self.n_way * (self.n_support + self.n_query), *obj.size()[2:]) for obj in x]
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
        # Embed the input data using the backbone network
        z_all = self.feature.forward(x)
        # Apply SelfOT transformation to the embedded data
        z_all = self.transform(z_all)
        # Reshape the transformed data for classification
        z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)

    return z_all

def set_forward(self, x, is_feature=False):
    """
    Performs a forward pass on the input data and computes classification scores.

    Parameters:
    - x (Tensor or list of Tensors): The input data.
    - is_feature (bool): Flag indicating whether the input data is already in feature form.

    Returns:
    - scores (Tensor): The classification scores for the query set.
    """
    # Parse the input data to extract or compute features
    z_all = self.parse_feature(x, is_feature)
    # Calculate the mass (sum of features) for each class in the support set
    n_tot = self.n_query+self.n_support
    mass = torch.stack([torch.sum(z_all[...,i*n_tot:i*n_tot+self.n_support], dim=2) for i in range(self.n_way)]).permute(1, 2, 0)
    # Compute classification scores for the query set
    scores = mass[:, self.n_support:, :].flatten(0, 1)

    return scores

def set_forward_loss(self, x):
    """
    Computes the loss for a forward pass.

    Parameters:
    - x (Tensor or list of Tensors): The input data.

    Returns:
    - loss (Tensor): The computed loss value for the forward pass.
    """
    # Create labels for the query set
    y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
    y_query = Variable(y_query.cuda())

    # Compute classification scores
    scores = self.set_forward(x)

    # Compute and return the loss
    return self.loss_fn(scores, y_query)

