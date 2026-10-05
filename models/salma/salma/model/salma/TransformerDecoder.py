from torch import nn, Tensor
from typing import Optional
import torch.nn.functional as F
import copy
import torch

class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer implementation taken from the DETR codebase.
    It was slightly cleaned but thee behavior remains the same.
    See https://github.com/facebookresearch/detr/blob/main/models/transformer.py

    It returns object queries by attending to themselves (through self-attention) and to 
    the encoder image features called memory (through cross-attention).
    
    Args:
        d_model (int): Dimension of the encoder embedding space.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Hidden dimension in the feedforward network.
        dropout (float): Dropout probability applied after attention and MLP.
        activation (str): Activation function in the feedforward block.
        normalize_before (bool): 
            - If True, normalize before each sublayer.
            - If False, normalize after each sublayer.

    Returns:
        tgt: (num_queries, batch_size, d_model)
            The updated query embeddings after one decoder layer.
    """
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.1,
                 activation: str = "relu", 
                 normalize_before: bool = False):
        
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, 
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, 
                query_pos: Optional[Tensor] = None
                )-> Tensor :
        
        x = self.norm1(tgt) if self.normalize_before else tgt

        # Self Attention Stage, hence keys are equal to queries.
        q = k = self.with_pos_embed(x, query_pos)
        tgt2 = self.self_attn(q, k, value=x, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt) if not self.normalize_before else tgt

        x = self.norm2(tgt) if self.normalize_before else tgt

        # Cross Attention Stage
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(x, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt) if not self.normalize_before else tgt

        # FFN
        x = self.norm3(tgt) if self.normalize_before else tgt
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt) if not self.normalize_before else tgt

        return tgt
    

class TransformerDecoder(nn.Module):
    """
    Transformer decoder implementation taken from the DETR codebase.
    See https://github.com/facebookresearch/detr/blob/main/models/transformer.py

    This class simply uses multiple TransformerDecoderLayer in a row.
    """

    def __init__(self, 
                 decoder_layer: TransformerDecoderLayer, 
                 num_layers: int, 
                 norm: Optional[nn.LayerNorm] = None, 
                 return_intermediate: Optional[bool] = False):
        
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation: str = "relu") -> Tensor:
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


