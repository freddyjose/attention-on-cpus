# attention_pytorch.py
# Copied from ComfyUI: https://github.com/comfyanonymous/ComfyUI/blob/main/comfy/ldm/modules/attention.py
# fetched on June 26th 2025, ideally this should be updated manually if there is a change
# Implements scaled dot-product attention for benchmarking against C++ DLL.

import torch
import math
from einops import rearrange

SDP_BATCH_LIMIT = 2**31

def attention_pytorch(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    if SDP_BATCH_LIMIT >= b:
        print("no batching")
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        if not skip_output_reshape:
            out = (
                out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            )
    else:
        print("batched execution")
        out = torch.empty((b, q.shape[2], heads * dim_head), dtype=q.dtype, layout=q.layout, device=q.device)
        for i in range(0, b, SDP_BATCH_LIMIT):
            m = mask
            if mask is not None:
                if mask.shape[0] > 1:
                    m = mask[i : i + SDP_BATCH_LIMIT]

            out[i : i + SDP_BATCH_LIMIT] = torch.nn.functional.scaled_dot_product_attention(
                q[i : i + SDP_BATCH_LIMIT],
                k[i : i + SDP_BATCH_LIMIT],
                v[i : i + SDP_BATCH_LIMIT],
                attn_mask=m,
                dropout_p=0.0, is_causal=False
            ).transpose(1, 2).reshape(-1, q.shape[2], heads * dim_head)
    return out
