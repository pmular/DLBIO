from methods.transform import SelfAttention
from torch.autograd import Variable
from methods.protonet import ProtoNet



class ProtoNetAttention(ProtoNet):
    """
    Extension of the ProtoNet model that incorporates a self-attention mechanism.
    This allows the model to weigh different parts of the input differently, potentially
    improving its ability to focus on the most relevant features for the few-shot task.

    Attributes:
    - self_attention (SelfAttention): A self-attention layer that is applied to the features
                                      extracted by the backbone model.

    Parameters:
    - backbone (nn.Module): The neural network used to extract features from input data.
    - n_way (int): The number of classes in the few-shot classification task.
    - n_support (int): The number of examples per class in the support set.
    """

    def __init__(self, backbone, n_way, n_support):
        super(ProtoNetAttention, self).__init__(backbone, n_way, n_support)
        self.self_attention = SelfAttention()

    def parse_feature(self, x, is_feature):
        """
        Extracts features from the input data, processes them through the self-attention
        mechanism, and separates support and query sets for few-shot classification.

        Parameters:
        - x (Tensor or list of Tensors): The input data, which can either be raw data or pre-extracted features.
        - is_feature (bool): A flag indicating whether the input x is already in feature form.

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
            # If input is raw data, reshape it and pass it through the backbone and self-attention layers
            if isinstance(x, list):
                x = [obj.contiguous().view(self.n_way * (self.n_support + self.n_query), *obj.size()[2:]) for obj in x]
            else:
                x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            # Embed the input data using the backbone network
            z_all = self.feature.forward(x)
            # Apply self-attention to the embedded data
            z_all = self.self_attention(z_all)
            # Reshape the features to separate support and query sets
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)

        # Separate the support and query features
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query

