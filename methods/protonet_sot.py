from torch.autograd import Variable
from methods.transform import SelfOT
from methods.protonet import ProtoNet



class ProtoNetSOT(ProtoNet):
    """
    Extension of the ProtoNet model that applies a Self Organizing Transformer (SelfOT)
    for transforming the embedding space to improve the few-shot learning performance.

    Attributes:
    - transform (SelfOT): An instance of SelfOT for transforming the feature space.

    Parameters:
    - backbone (nn.Module): The neural network model used to extract features from input data.
    - n_way (int): The number of classes in the few-shot classification task.
    - n_support (int): The number of examples per class in the support set.
    """

    def __init__(self, backbone, n_way, n_support):
        super(ProtoNetSOT, self).__init__(backbone, n_way, n_support)
        self.transform = SelfOT()

    def parse_feature(self, x, is_feature):
        """
        Extracts and transforms features from the input data using the backbone network
        and SelfOT, then separates the support and query sets for few-shot classification.

        Parameters:
        - x (Tensor or list of Tensors): The input data, which can either be raw data or pre-extracted features.
        - is_feature (bool): A flag indicating whether the input 'x' is already in feature form.

        Returns:
        - z_support (Tensor): The processed features corresponding to the support set.
        - z_query (Tensor): The processed features corresponding to the query set.
        """
        
        # Convert input to Variables and move to the correct device
        if isinstance(x, list):
            x = [Variable(obj.to(self.device)) for obj in x]
        else:
            x = Variable(x.to(self.device))

        # If input is already features, use it directly
        if is_feature:
            z_all = x
        else:
            # If input is raw data, reshape it and pass it through the backbone and SelfOT layers
            if isinstance(x, list):
                x = [obj.contiguous().view(self.n_way * (self.n_support + self.n_query), *obj.size()[2:]) for obj in x]
            else:
                x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            # Embed the input data using the backbone network
            z_all = self.feature.forward(x)
            # Apply SelfOT transformation to the embedded data
            z_all = self.transform(z_all)
            # Reshape the features to separate support and query sets
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)

        # Separate the support and query features
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query




 

