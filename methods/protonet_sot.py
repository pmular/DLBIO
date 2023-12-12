# This code is modified from https://github.com/jakesnell/prototypical-networks 

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
from methods.transform import SelfOT
from methods.meta_template import MetaTemplate
from methods.protonet import ProtoNet


class ProtoNetSOT(ProtoNet):
    def __init__(self, backbone, n_way, n_support, self_ot):
        super(ProtoNetSOT, self).__init__(backbone, n_way, n_support)
        self.transform = SelfOT(reg=self_ot.reg, diag_weight=self_ot.diag_weight, tol=self_ot.tol, tau=self_ot.tau)

    
    def parse_feature(self, x, is_feature):
        if isinstance(x, list):
            x = [Variable(obj.to(self.device)) for obj in x]
        else: x = Variable(x.to(self.device))
        if is_feature:
            z_all = x
        else:
            if isinstance(x, list):
                x = [obj.contiguous().view(self.n_way * (self.n_support + self.n_query), *obj.size()[2:]) for obj in x]
            else: x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all = self.feature.forward(x) # embedding with backbone
            z_all = self.transform(z_all)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)

        return z_all

    
    def set_forward(self, x, is_feature=False):
        
        z_all = self.parse_feature(x, is_feature)
        n_tot = self.n_query+self.n_support
        mass = torch.stack([torch.sum(z_all[...,i*n_tot:i*n_tot+self.n_support], dim=2) for i in range(self.n_way)]).permute(1, 2, 0)
        scores = mass[:, self.n_support:, :].flatten(0, 1)

        return scores


 


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
