# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerConfig
from fairseq.modules import MultiheadAttention, transformer_layer

logger = logging.getLogger(__name__)


def cumprod(x, reverse=False, exclusive=False):
  """cumulative product."""
  if reverse:
    x = x.flip([-1])

  if exclusive:
    x = F.pad(x[:, :, :-1], (1, 0), value=1)

  cx = x.cumprod(-1)

  if reverse:
    cx = cx.flip([-1])
  return cx


def cumsum(x, reverse=False, exclusive=False):
  """cumulative sum."""
  bsz, _, length = x.size()
  device = x.device
  if reverse:
    if exclusive:
      w = torch.ones([bsz, length, length], device=device).tril(-1)
    else:
      w = torch.ones([bsz, length, length], device=device).tril(0)
    cx = torch.bmm(x, w)
  else:
    if exclusive:
      w = torch.ones([bsz, length, length], device=device).triu(1)
    else:
      w = torch.ones([bsz, length, length], device=device).triu(0)
    cx = torch.bmm(x, w)
  return cx


def cummin(x, reverse=False, exclusive=False, max_value=1e9):
  """cumulative min."""
  if reverse:
    if exclusive:
      x = F.pad(x[:, :, 1:], (0, 1), value=max_value)
    x = x.flip([-1]).cummin(-1)[0].flip([-1])
  else:
    if exclusive:
      x = F.pad(x[:, :, :-1], (1, 0), value=max_value)
    x = x.cummin(-1)[0]
  return x


@register_model("grammarformer")
class GrammarFormer(TransformerModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @staticmethod
    def add_args(parser):
        super(GrammarFormer, GrammarFormer).add_args(parser)
        parser.add_argument("--conv-size", type=int, metavar="N", default=9,
            help="convolution kernel size for parser")
        parser.add_argument("--n-parser-layers", type=int, metavar="N", default=3,
            help="number of parsing layers")
        parser.add_argument("--n-mask-layers", type=int, metavar="N", default=1,
            help="number of layers being masked")
        parser.add_argument("--weight-act", type=str, metavar="STR", default="softmax",
            help="relations distribution activation function")
        parser.add_argument("--lm-dropout", type=float, metavar="D", default=0.1,
            help="dropout for lm layer")
        parser.add_argument("--mask-rate", type=float, metavar="D", default=0.15,
            help="mask rate for mlm")
        parser.add_argument("--lm-scaler", type=float, metavar="D", default=0.47,
            help="weight for lm_loss")
        parser.add_argument("--n-lm-layers", type=int, metavar="D", default=3,
            help="number of encoder layers used mlm")
        parser.add_argument("--bpe-dim", type=int, metavar="D", default=256,
            help="dimension of bpe embedding to be concatenated")

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return GrammarFormerEncoder(cfg, src_dict, embed_tokens)

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_bpe=None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, src_bpe=src_bpe
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        extra = {
            "use_lm": True,
            "lm_loss": encoder_out["lm_loss"]
        }
        return (decoder_out, extra)


class GrammarFormerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens, return_fc=False):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            return_fc=return_fc,
        )
        self.nlayers = args.encoder.layers
        self.nhead = args.encoder.attention_heads

        self.conv_size = args.conv_size
        self.n_parser_layers = args.n_parser_layers
        self.weight_act = args.weight_act
        self.lm_scaler = args.lm_scaler
        # self.n_lm_layers = args.n_lm_layers

        embed_dim = args.encoder.embed_dim
        self.embed_bpe = Embedding(3, args.bpe_dim, padding_idx=self.padding_idx)
        self.embed_proj = nn.Sequential(
            nn.Linear(args.bpe_dim, embed_dim),
            nn.LayerNorm(embed_dim, elementwise_affine=False),
            nn.Tanh())

        self.mask_bernoulli = torch.distributions.Bernoulli(args.mask_rate)

        # self.parser_emb = nn.Linear(embed_dim, embed_dim)
        self.parser_layers = nn.ModuleList([
            nn.Sequential(Conv1d(embed_dim, args.conv_size),
                        nn.LayerNorm(embed_dim, elementwise_affine=False),
                        nn.Tanh()) for i in range(args.n_parser_layers)])

        # self.parser_encoder = transformer_layer.TransformerEncoderLayerBase(args)
        self.parser_attention = MultiheadAttention(
            embed_dim,
            num_heads=1,
            dropout=0,
            self_attention=True)

        self.distance_ff = nn.Sequential(
            Conv1d(embed_dim, 2),
            nn.LayerNorm(embed_dim, elementwise_affine=False), nn.Tanh(),
            nn.Linear(embed_dim, 1))

        self.height_ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim, elementwise_affine=False), nn.Tanh(),
            nn.Linear(embed_dim, 1))

        # relations that are used to compute self attention
        self.relations = ('head', 'child')
        n_rel = len(self.relations)
        self._rel_weight = nn.Parameter(torch.zeros((args.n_mask_layers, self.nhead, n_rel)))
        self._rel_weight.data.normal_(0, 0.1)

        self._scaler = nn.Parameter(torch.zeros(2))
        # import pdb
        # pdb.set_trace()
        
        # self.lm_encoder = transformer_layer.TransformerEncoderLayerBase(args)
        self.lm_layer = nn.Sequential(
            nn.LayerNorm(embed_dim, elementwise_affine=False),
            nn.Dropout(args.lm_dropout))

        self.lm_output = nn.Linear(embed_dim, len(dictionary))
        self.lm_output.weight = self.embed_tokens.weight
        
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=self.padding_idx, reduction='sum')

        self.mask_idx = 0
        

    def mask_data(self, data, bpe=None):
        """randomly mask input sequence."""
        mask = self.mask_bernoulli.sample(data.shape).to(data.device.index).bool()
        mask = mask * (data != self.padding_idx)
        targets = data.masked_fill(~mask, self.padding_idx) # [1,1,1,4,1,6,1]
        data = data.masked_fill(mask, self.mask_idx) # [9,3,4,0,3,0,8]
        if bpe is not None:
            bpe = bpe.masked_fill(mask, self.mask_idx)
            return data, targets, bpe
        else:
            return data, targets

    @property
    def scaler(self):
        return self._scaler.exp()

    @property
    def rel_weight(self):
        if self.weight_act == 'sigmoid':
            return torch.sigmoid(self._rel_weight)
        elif self.weight_act == 'softmax':
            return torch.softmax(self._rel_weight, dim=-1)

    def parse(self, x, pos, h, bpe_embedding=None):
        """Parse input sentence.

        Args:
            x: input tokens (required).
            pos: position for each token (optional).
            h: embeddings od input tokens
        Returns:
            distance: syntactic distance
            height: syntactic height
        """

        mask = (x != self.padding_idx)
        mask_shifted = F.pad(mask[:, 1:], (0, 1), value=0)

        for i in range(self.n_parser_layers):
            h = h.masked_fill(~mask[:, :, None], 0)
            h = self.parser_layers[i](h)
         
        key_padding_mask = x.eq(self.padding_idx)
        h = h.transpose(0, 1)
        h, _ = self.parser_attention(
            query=h,
            key=h,
            value=h,
            key_padding_mask=key_padding_mask,
            )
        h = h.transpose(0, 1)

        if bpe_embedding is not None:
            bpe_embedding = bpe_embedding.masked_fill(~mask[:, :, None], 0)
            bpe = self.embed_proj(bpe_embedding)
            h = h + bpe

        # height: batch_size * length
        height = self.height_ff(h).squeeze(-1)
        # height = 1. / (1. + torch.exp(0 - height))
        height.masked_fill_(~mask, -1e9)

        # distance: batch_size * length
        distance = self.distance_ff(h).squeeze(-1)
        distance.masked_fill_(~mask_shifted, 1e9)

        # Calbrating the distance and height to the same level
        length = distance.size(1)
        # height_max: batch_size * length * length
        height_max = height[:, None, :].expand(-1, length, -1)
        height_max = torch.cummax(
            height_max.triu(0) - torch.ones_like(height_max).tril(-1) * 1e9,
            dim=-1)[0].triu(0)

        margin_left = torch.relu(
            F.pad(distance[:, :-1, None], (0, 0, 1, 0), value=1e9) - height_max)
        margin_right = torch.relu(distance[:, None, :] - height_max)

        margin = torch.where(margin_left > margin_right, margin_right,margin_left).triu(0)

        margin_mask = torch.stack([mask_shifted] + [mask] * (length - 1), dim=1)
        margin.masked_fill_(~margin_mask, 0)
        # margin = margin.max()

        # distance = distance - margin
        margin = margin.view(x.shape[0], -1)
        margin = margin.max(dim=-1)[0]

        distance = distance - margin.unsqueeze(1).expand_as(distance)
        # distance[:, -2] = distance[:, -2] - 1e9
        # import pdb
        # pdb.set_trace()
        return distance, height

    def compute_block(self, distance, height):
        """Compute constituents from distance and height."""

        # beta_logits: batch_size * length * length
        beta_logits = (distance[:, None, :] - height[:, :, None]) * self.scaler[0]

        gamma = torch.sigmoid(-beta_logits)
        ones = torch.ones_like(gamma)

        # (*, i, j), j < i: probability that x_j belongs to constituent C(x_i)
        block_mask_left = cummin(
            gamma.tril(-1) + ones.triu(0), reverse=True, max_value=1)
        # (*, i, j), j < i: probability that x_j is the left boundary of constituent C(x_i)
        block_mask_left = block_mask_left - F.pad(
            block_mask_left[:, :, :-1], (1, 0), value=0)
        block_mask_left.tril_(0)

        block_mask_right = cummin(
            gamma.triu(0) + ones.tril(-1), exclusive=True, max_value=1)
        block_mask_right = block_mask_right - F.pad(
            block_mask_right[:, :, 1:], (0, 1), value=0)
        block_mask_right.triu_(0)

        # (*, i, j, k): probability that [j, k] == constituent C(x_i)
        block_p = block_mask_left[:, :, :, None] * block_mask_right[:, :, None, :]
        # (*, i, j): probability that x_j belongs to constituent C(x_i)
        block = cumsum(block_mask_left).tril(0) + cumsum(
            block_mask_right, reverse=True).triu(1)

        return block_p, block

    def compute_head(self, height):
        """Estimate head for each constituent."""

        _, length = height.size()
        head_logits = height * self.scaler[1]
        index = torch.arange(length, device=height.device)

        mask = (index[:, None, None] <= index[None, None, :]) * (
            index[None, None, :] <= index[None, :, None])
        head_logits = head_logits[:, None, None, :].repeat(1, length, length, 1)
        head_logits.masked_fill_(~mask[None, :, :, :], -1e9)

        # (*, i, j, k): probability that x_i is the parent of [j, k]
        # exp(h_i/u) / sum[j<=t<=k](exp(h_t/u)), j<=i<=k; 0, otherwise, i.e. out of the span
        head_p = torch.softmax(head_logits, dim=-1)

        return head_p

    def generate_mask(self, x, distance, height):
        """Compute head and cibling distribution for each token."""

        bsz, length = x.size()

        eye = torch.eye(length, device=x.device, dtype=torch.bool)
        eye = eye[None, :, :].expand((bsz, -1, -1))

        block_p, block = self.compute_block(distance, height)
        head_p = self.compute_head(height)
        # P_D = P_Pr * P_C
        head = torch.einsum('blij,bijh->blh', block_p, head_p)
        head = head.masked_fill(eye, 0)
        child = head.transpose(1, 2)
        cibling = torch.bmm(head, child).masked_fill(eye, 0)

        rel_list = []
        if 'head' in self.relations:
            rel_list.append(head)
        if 'child' in self.relations:
            rel_list.append(child)
        if 'cibling' in self.relations:
            rel_list.append(cibling)

        # rel: batch_szie * num_relations * length * length
        rel = torch.stack(rel_list, dim=1)

        # rel_weight: num_layers * num_heads * num_relations
        rel_weight = self.rel_weight

        dep = torch.einsum('lhr,brij->lbhij', rel_weight, rel)
        att_mask = dep.reshape(rel_weight.shape[0], bsz * self.nhead, length, length)
        # import pdb
        # pdb.set_trace()
        # allocated = att_mask.sum(dim = -1)
        # remained = torch.relu(torch.ones_like(allocated) - allocated)
        # diag = torch.eye(length, device=x.device).expand_as(att_mask)
        # diag_mask = remained.unsqueeze(-1).expand_as(att_mask).masked_fill(diag==0, 0)
        # att_mask = att_mask + diag_mask

        return att_mask, cibling, head, block

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        src_bpe=None
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings.

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings, src_bpe
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        src_bpe=None
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings.

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.training:
            if src_bpe is not None:
                src_tokens, targets, src_bpe = self.mask_data(src_tokens, src_bpe)
            else:
                src_tokens, targets = self.mask_data(src_tokens)
  
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        bpe_embedding = None
        if src_bpe is not None:
            bpe_embedding = self.embed_bpe(src_bpe)
            # h = torch.cat((x, bpe_embedding), dim=2)
            # h = h * (1 - encoder_padding_mask.unsqueeze(-1).type_as(h))

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # src_tokens: batch_size * length
        batch_size, length = src_tokens.size()
        pos = torch.arange(src_tokens.size(1))[None, :]

        distance, height = self.parse(src_tokens, pos, x, bpe_embedding)
        attn_mask, cibling, head, block = self.generate_mask(src_tokens, distance, height)
        # height_mask = torch.softmax(height, dim=-1)
        # height_mask = height_mask.unsqueeze(-1).repeat_interleave(height_mask.shape[-1],dim=-1)
        # height_mask = height_mask.repeat_interleave(self.nhead, dim=0)
      
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)

        # import pdb
        # pdb.set_trace()
        # encoder layers
        lm_loss = None
        for idx, layer in enumerate(self.layers):
            lr = layer(
                x, 
                encoder_padding_mask=encoder_padding_mask if has_pads else None, 
                weight_mask=attn_mask[idx] if idx < attn_mask.shape[0] else None
            )

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)
        
        if self.training:# and (idx + 1 == self.n_lm_layers):
            # lm_out = self.lm_encoder(x, encoder_padding_mask=encoder_padding_mask if has_pads else None)
            lm_out = self.lm_layer(x.transpose(0, 1))
            lm_out = self.lm_output(lm_out)
            lm_loss = self.lm_criterion(lm_out.view(batch_size * length, -1), targets.reshape(-1))

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
            "lm_loss": lm_loss
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
            "lm_loss": encoder_out["lm_loss"]
        }


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


class Conv1d(nn.Module):
  """1D convolution layer."""

  def __init__(self, hidden_size, kernel_size, dilation=1):
    """Initialization.

    Args:
      hidden_size: dimension of input embeddings
      kernel_size: convolution kernel size
      dilation: the spacing between the kernel points
    """
    super(Conv1d, self).__init__()

    if kernel_size % 2 == 0:
      padding = (kernel_size // 2) * dilation
      self.shift = True
    else:
      padding = ((kernel_size - 1) // 2) * dilation
      self.shift = False
    self.conv = nn.Conv1d(
        hidden_size,
        hidden_size,
        kernel_size,
        padding=padding,
        dilation=dilation)

  def forward(self, x):
    """Compute convolution.

    Args:
      x: input embeddings
    Returns:
      conv_output: convolution results
    """

    if self.shift:
      return self.conv(x.transpose(1, 2)).transpose(1, 2)[:, 1:]
    else:
      return self.conv(x.transpose(1, 2)).transpose(1, 2)


@register_model_architecture("grammarformer", "grammarformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)


@register_model_architecture("grammarformer", "grammarformer_iwslt_de_en")
def grammarformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture("grammarformer", "grammarformer_wmt_en_de")
def grammarformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("grammarformer", "grammarformer_vaswani_wmt_en_de_big")
def grammarformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture("grammarformer", "grammarformer_vaswani_wmt_en_fr_big")
def grammarformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, "dropout", 0.1)
    grammarformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("grammarformer", "grammarformer_wmt_en_de_big")
def grammarformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    grammarformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture("grammarformer", "grammarformer_wmt_en_de_big_t2t")
def grammarformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    grammarformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("grammarformer", "grammarformer_nc11_de_en")
def grammarformer_nc11_de_en(args):
    grammarformer_iwslt_de_en(args)


@register_model_architecture("grammarformer", "grammarformer_aspec_ch_ja")
def grammarformer_aspec_ch_ja(args):
    base_architecture(args)


@register_model_architecture("grammarformer", "grammarformer_wmt17_en_tr")
def grammarformer_wmt17_en_tr(args):
    grammarformer_iwslt_de_en(args)


@register_model_architecture("grammarformer", "grammarformer_wmt19_en_kk")
def grammarformer_wmt19_en_kk(args):
    grammarformer_iwslt_de_en(args)

