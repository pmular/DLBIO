class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
       
        # ToDo: initialize K, Q, V
        #x = [n, d]
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)

        # ToDo: add a dropout layer
        # Hint: using config.attention_probs_dropout_prob as the dropout probability
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        # The parameter encoder_hidden_states and encoder_attention_mask is for cross-attention. 
        # We do not use them in this homework.

        # ToDo: get the key, query, and value from the hidden_states
        mixed_key_layer = self.key(x)  n x d
        mixed_query_layer = self.query(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # ToDo: transpose K, Q, V to get the score
        # Hint: using self.transpose_for_scores
       

        # ToDo: Get the raw attention score
        # Hint: Lecture 05 transformers - Slide 23 - the part within the softmax  [5, 20, 1000] -> [100, 1000] -> [100, 100] -> [5, 20, 100]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) n x n    #[5, 5, 100], [5, 15, 100]
        
        # You do not need to change this part.
        # Explanation of attention_mask: https://lukesalamone.github.io/posts/what-are-attention-masks/
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # ToDo: Normalize the attention scores to probabilities.
        # Hint: 
        # 1. Lecture 05 transformers - Slide 23 - using softmax to get the probability
        # 2. Use self.dropout to do the dropout

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # You do not need to change this part.
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # ToDo: Multiply each value by the score
        # Hint: 
        # 1. Lecture 05 transformers - Slide 23 - getting the final result
        # 2. Permuting the result to the correct shape. If you do not know what should be the correct shape, you can print the shape of the tensors.
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).reshape(hidden_states.shape)
        # Get the output
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs