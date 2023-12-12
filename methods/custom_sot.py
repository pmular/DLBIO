import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from methods.transform import SelfOT
from methods.meta_template import MetaTemplate


class CustomSOT(MetaTemplate):
    def __init__(self, backbone, n_way, n_support, self_ot):
        super(CustomSOT, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
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

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )

