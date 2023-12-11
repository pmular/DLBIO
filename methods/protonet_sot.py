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
    def __init__(self, backbone, n_way, n_support, cfg):
        super(ProtoNetSOT, self).__init__(backbone, n_way, n_support)
        self.transform = SelfOT(reg=cfg.self_ot.reg, diag_weight=cfg.self_ot.diag_weight, tol=cfg.self_ot.tol, tau=cfg.self_ot.tau)

    
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
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query


class ProtoNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(ProtoNet, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False):
        
        z_support, z_query = self.parse_feature(x, is_feature)
        
        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # centroids
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )


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
