import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.attention import SDPBackend, sdpa_kernel


class TransformerClassifier(nn.Module):
    def __init__(self, inputfeature_dim: int, num_classes: int, num_heads: int, 
                 embed_dim: int, num_layers: int, use_flash_attention: bool=False, is_causal: bool=True, dropout:float=0.0):
        super(TransformerClassifier, self).__init__()
        assert embed_dim % num_heads == 0
        self.embedding = nn.Linear(inputfeature_dim, embed_dim)
        if use_flash_attention:
            encoder_layer = CustomTransformerEncoderLayer(
                embed_dim,
                num_heads,
                dim_feedforward=embed_dim * 2,
                dropout=dropout,
                )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,
                dropout=dropout,
                # batch_first: If "True", then the input and output tensors are provided
                # as (batch, seq, feature). Default: "False" (seq, batch, feature).
                batch_first=True,
            )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes
        
    def forward(self,x):
        x = self.embedding(x)

        padding_mask = (x.sum(dim=-1) == 0)
        
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        # x = self.transformer_encoder(x)
        x = self.classifier(x)
        return x

class CustomTransformerEncoderLayer(nn.Module):
    '''
    Taken and adapted from pytorch's implementation of TransformerDecoderLayer:
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer.forward
    '''
    __constants__ = ['norm_first']

    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int, bias: bool, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 1e-5, norm_first: bool = False, dropout:float=0.0, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.self_attn = CausalSelfAttention(num_heads, embed_dim, bias, dropout=dropout)
        # Implementation of Feedforward model 
        self.linear1 = nn.Linear(embed_dim, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim, bias=bias, **factoray_kwargs)
       
        self.norm_first = norm_first 
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(self, src: Tensor, src_mask: Optional[Tensor]=None, src_key_padding_mask: Optional[Tensor]=None):
       
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

class CausalSelfAttention(nn.Module):
    '''
    Taken from the pytorch tutorial on SDPA:
    https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
    '''

    def __init__(self, num_heads: int, embed_dimension: int, bias: bool, is_causal: bool=True, dropout:float=0.0):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        # Perform causal masking
        self.is_causal = is_causal

    def forward(self, x):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

        # input is in batch_size, sequence_length, embedding dimensionality
        # this is equivalent to batch_first = True for TransformerEncoderLayer
        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        if self.training:
            dropout = self.dropout
            is_causal = self.is_causal
        else:
            dropout = 0.0
            is_causal = False

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            y = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout, is_causal=is_causal)

        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))
        return y
