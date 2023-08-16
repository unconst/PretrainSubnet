# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cuda as cuda
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Attention

# patch GPT2Attention to use flash_sdp, disable it when doing the inference
def _attn_wrapper(self, query, key, value, attention_mask=None, head_mask=None):
    if head_mask is not None:
        raise NotImplementedError("head_mask is not implemented for flash_sdp")
    is_causal = attention_mask is None
    with cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False,
    ):
        attn_out = F.scaled_dot_product_attention(
            query=query.half(),
            key=key.half(),
            value=value.half(),
            is_causal=is_causal,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p,
        ).float()
    return attn_out, None

def make_model( config ):
    
    # Build the normal gpt2 model.
    if config.model_type == 'gpt2':
        model = GPT2LMHeadModel(GPT2Config(n_layer = config.n_layer, n_head = config.n_head))

    if config.model_type == 'tiny_gpt2':
        model = GPT2LMHeadModel(GPT2Config(n_layer = 1, n_head = 1))
        
    # Buidl the long gpt model.
    elif config.model_type == 'long_gpt':
        GPT2Attention._attn = _attn_wrapper
        model.config.update(
            dict(
                n_ctx = config.sl,
                n_positions = config.sl,
            )
        )
        # patch model embeddings
        emb = model.transformer.wpe.weight.data
        wpe = nn.Embedding( config.sl, emb.shape[1])
        wpe.weight.data = emb.repeat( config.sl // emb.shape[0], 1)
        model.transformer.wpe = wpe

        # also increase mask size
        for block in model.transformer.h:
            block.attn.bias.data = (
                torch.tril(torch.ones(( config.sl, config.sl), dtype=torch.bool))
                .view(1, 1, config.sl, config.sl)
                .cuda()
            )
    else:
        raise RuntimeError(f"{config.model_type} is not a valid type.")
    return model