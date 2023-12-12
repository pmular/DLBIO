import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
from methods.transform import SelfOT
from methods.meta_template import MetaTemplate
from methods.protonet import ProtoNet


class ProtoNetSOT(ProtoNet):
    def __init__(self, backbone, n_way, n_support):
        super(ProtoNetSOT, self).__init__(backbone, n_way, n_support)
        self.transform = SelfOT()

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



 

