from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import transformers.models.gpt2.modeling_gpt2 as modeling_gpt2
from config import CONFIG

# Overwritable functions for interacting with drop-in

def consolidate_kv(key, value):
    return key, value


def record_attn_vars(layer_idx, query, key, value, unnormalized_attn, final_attn):
    pass


class Globals:
    def __init__(self):
        self.new_params = []
        self.outputs = []
        self.attention_svd = 0

# From huggingface

class ScaleHead(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.zeros(nx, nf))
        self.bias = nn.Parameter(torch.ones(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class GPT2AttentionDropIn(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)
        if CONFIG.scale_v:
            print("Training scale factor function per value vector")
            self.scale_v = ScaleHead(self.num_heads, self.embed_dim)  # scaling factor per v
            GLOBALS.new_params.extend(self.scale_v.parameters())

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """Shapes:
        query:          torch.Size([batch, n heads, L, d])
        key:            torch.Size([batch, n heads, L, d])
        value:          torch.Size([batch, n heads, L, d])
        attention_mask: torch.Size([batch, 1, 1, L])
        """

        ############################
        # consolidate kv cache #####
        if CONFIG.do_consolidate:
            key, value = consolidate_kv(key, value)
        ############################

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        unnormalized_attn = attn_weights.clone()

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        ############################
        if GLOBALS.attention_svd > 0:
            s_, v_, d_ = torch.svd(attn_weights)
            attn_weights = s_[...,:GLOBALS.attention_svd] @ \
                torch.diag_embed(
                    v_[...,:GLOBALS.attention_svd].reshape(-1, GLOBALS.attention_svd)
                ).reshape(*
                    s_.shape[:-2], GLOBALS.attention_svd, GLOBALS.attention_svd
                ) @ d_[...,:GLOBALS.attention_svd].transpose(-1, -2)
        ############################

        attn_output = torch.matmul(attn_weights, value)

        ############################
        # record data for analysis #
        record_attn_vars(self.layer_idx, query, key, value, unnormalized_attn, attn_weights)
        ############################

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        #######################
        # scale v #############

        # (batch, head, seq_length, head_features)

        if CONFIG.scale_v:
            v_scale = self.scale_v(hidden_states).permute(0, 2, 1).unsqueeze(-1)
            value = value * v_scale

        #######################

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # if self.reorder_and_upcast_attn:
        #     attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        # else:

        # replace kv cache with updated version
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


GLOBALS = Globals()
OriginalAttention = GPT2Attention
modeling_gpt2.GPT2Attention = GPT2AttentionDropIn
