"""KV-cached incremental inference for PriorDiscrete.

`PriorDiscrete.forward` recomputes the full transformer stack over the whole
context every step, which makes autoregressive generation O(S^2) and caps the
usable context length in realtime. This module reimplements the *same* forward
math (post-LN transformer, ReLU FFN, the `* sqrt(d_model)` output scale, the
per-codebook embedding offsets and absolute positional encoding) but keeps a
per-layer rolling K/V cache so each new token costs O(S) instead of O(S^2).

It is an inference-only module: weights are copied from a trained `PriorDiscrete`
(which keeps using `nn.TransformerEncoder` for training). The single entry point
`decode(tokens)` appends tokens to the cache and returns their logits; it serves
cold start (`decode([START])`), incremental stepping (`decode([tok])`) and
re-priming on window slide (`reset()` then `decode(retained_ids)`).

Positions are absolute within the window (pe[len .. len+S]); when the window is
about to overflow `max_len`, the caller re-primes (reset + decode of the retained
tail) so positions stay in `[0, max_len)` exactly as during training.

Written to be TorchScript-scriptable (typed, tensor-ops only).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class _KVLayer(nn.Module):
    """One post-LN transformer encoder layer with a rolling K/V cache (batch=1)."""

    def __init__(self, src_layer: nn.TransformerEncoderLayer, d_model: int, nhead: int, max_len: int):
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = int(d_model // nhead)
        self.max_len = int(max_len)

        # Copy trained weights (no grad needed at inference) as buffers.
        sa = src_layer.self_attn
        self.register_buffer("in_proj_weight", sa.in_proj_weight.detach().clone())
        self.register_buffer("in_proj_bias", sa.in_proj_bias.detach().clone())
        self.register_buffer("out_proj_weight", sa.out_proj.weight.detach().clone())
        self.register_buffer("out_proj_bias", sa.out_proj.bias.detach().clone())
        self.register_buffer("lin1_weight", src_layer.linear1.weight.detach().clone())
        self.register_buffer("lin1_bias", src_layer.linear1.bias.detach().clone())
        self.register_buffer("lin2_weight", src_layer.linear2.weight.detach().clone())
        self.register_buffer("lin2_bias", src_layer.linear2.bias.detach().clone())
        self.register_buffer("norm1_weight", src_layer.norm1.weight.detach().clone())
        self.register_buffer("norm1_bias", src_layer.norm1.bias.detach().clone())
        self.register_buffer("norm2_weight", src_layer.norm2.weight.detach().clone())
        self.register_buffer("norm2_bias", src_layer.norm2.bias.detach().clone())

        # Rolling caches: [1, nhead, max_len, head_dim]
        self.register_buffer("k_cache", torch.zeros(1, self.nhead, self.max_len, self.head_dim))
        self.register_buffer("v_cache", torch.zeros(1, self.nhead, self.max_len, self.head_dim))

    def decode(self, x: torch.Tensor, off: int, attn_mask: torch.Tensor) -> torch.Tensor:
        """x: [S, 1, D] for positions [off, off+S). attn_mask: [S, off+S] additive.
        Returns [S, 1, D]. Writes the new K/V into the cache at [off:off+S]."""
        s = x.shape[0]
        # in-proj -> q,k,v each [S,1,D]
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # [S,1,D] -> [1, nhead, S, head_dim]
        q = q.view(s, self.nhead, self.head_dim).permute(1, 0, 2).unsqueeze(0)
        k = k.view(s, self.nhead, self.head_dim).permute(1, 0, 2).unsqueeze(0)
        v = v.view(s, self.nhead, self.head_dim).permute(1, 0, 2).unsqueeze(0)

        # Write into the cache, then read the full prefix [0, off+S).
        self.k_cache[:, :, off:off + s, :] = k
        self.v_cache[:, :, off:off + s, :] = v
        k_full = self.k_cache[:, :, 0:off + s, :]
        v_full = self.v_cache[:, :, 0:off + s, :]

        attn = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=attn_mask)
        # [1, nhead, S, head_dim] -> [S,1,D]
        attn = attn.squeeze(0).permute(1, 0, 2).reshape(s, 1, self.d_model)
        attn = F.linear(attn, self.out_proj_weight, self.out_proj_bias)

        # post-LN
        x = F.layer_norm(x + attn, [self.d_model], self.norm1_weight, self.norm1_bias, 1e-5)
        ff = F.linear(F.relu(F.linear(x, self.lin1_weight, self.lin1_bias)), self.lin2_weight, self.lin2_bias)
        x = F.layer_norm(x + ff, [self.d_model], self.norm2_weight, self.norm2_bias, 1e-5)
        return x


class KVCachedPrior(nn.Module):
    """Incremental, KV-cached equivalent of PriorDiscrete.forward (batch=1)."""

    def __init__(self, prior):
        super().__init__()
        self.num_codebooks = int(prior.num_codebooks)
        self.codebook_size = int(prior.codebook_size)
        self.embedding_dim = int(prior._embedding_dim)
        self.d_model = int(prior._d_model)
        self.max_len = int(prior._max_len)
        self.start_id = int(prior.start_token_id)
        self.vocab_per_codebook = int(prior._vocab_per_codebook)
        self.sqrt_d = float(math.sqrt(self.d_model))

        self.register_buffer("embedding_weight", prior._embedding.weight.detach().clone())
        self.register_buffer("codebook_offsets", prior._codebook_offsets.detach().clone().long())
        # pe buffer is [max_len, 1, embedding_dim]; keep as [max_len, embedding_dim].
        self.register_buffer("pe", prior._positional_encoding.pe.detach().clone().squeeze(1))
        self.register_buffer("fc_weight", prior._fc.weight.detach().clone())
        self.register_buffer("fc_bias", prior._fc.bias.detach().clone())

        # Territory conditioning vectors [num_territories, d_model] (zeros row when disabled).
        self.num_territories = int(getattr(prior, "_num_territories", 0))
        self.has_territory = bool(getattr(prior, "_territory_embedding", None) is not None)
        if self.has_territory:
            self.register_buffer("territory_weight", prior._territory_embedding.weight.detach().clone())
        else:
            self.register_buffer("territory_weight", torch.zeros(1, self.d_model))

        # Envelope (LFO) conditioning projection, mapped from the trained _cond_proj.
        self.cond_dim = int(getattr(prior, "_cond_dim", 0))
        self.has_cond = bool(getattr(prior, "_cond_proj", None) is not None)
        if self.has_cond:
            self.register_buffer("cond_proj_weight", prior._cond_proj.weight.detach().clone())
            self.register_buffer("cond_proj_bias", prior._cond_proj.bias.detach().clone())
        else:
            self.register_buffer("cond_proj_weight", torch.zeros(self.d_model, 1))
            self.register_buffer("cond_proj_bias", torch.zeros(self.d_model))

        # Style conditioning: input-additive projection + per-layer FiLM, mapped from the trained model.
        # At realtime the style code comes from the XY pad (no encoder needed in the loop).
        self.style_dim = int(getattr(prior, "_style_dim", 0))
        self.has_style = bool(getattr(prior, "_style_proj", None) is not None and self.style_dim > 0)
        if self.has_style:
            self.register_buffer("style_proj_weight", prior._style_proj.weight.detach().clone())
            self.register_buffer("style_proj_bias", prior._style_proj.bias.detach().clone())
            fw = torch.stack([l.weight.detach().clone() for l in prior._style_film], 0)  # [L, 2D, style_dim]
            fb = torch.stack([l.bias.detach().clone() for l in prior._style_film], 0)     # [L, 2D]
            self.register_buffer("style_film_weight", fw)
            self.register_buffer("style_film_bias", fb)
        else:
            self.register_buffer("style_proj_weight", torch.zeros(self.d_model, 1))
            self.register_buffer("style_proj_bias", torch.zeros(self.d_model))
            self.register_buffer("style_film_weight", torch.zeros(1, 2 * self.d_model, 1))
            self.register_buffer("style_film_bias", torch.zeros(1, 2 * self.d_model))

        # WS4 joint codebook depth head, mapped from the trained model (zeros when independent).
        self.is_joint = bool(getattr(prior, "_joint", False))
        if self.is_joint:
            self.register_buffer("depth_pos_weight", prior._depth_pos.weight.detach().clone())
            self.register_buffer("depth_token_embed_weight", prior._depth_token_embed.weight.detach().clone())
            self.register_buffer("depth_mlp_weight", prior._depth_mlp.weight.detach().clone())
            self.register_buffer("depth_mlp_bias", prior._depth_mlp.bias.detach().clone())
            self.register_buffer("depth_fc_weight", prior._depth_fc.weight.detach().clone())
            self.register_buffer("depth_fc_bias", prior._depth_fc.bias.detach().clone())
        else:
            self.register_buffer("depth_pos_weight", torch.zeros(self.num_codebooks, self.d_model))
            self.register_buffer("depth_token_embed_weight", torch.zeros(self.num_codebooks * self.codebook_size, self.d_model))
            self.register_buffer("depth_mlp_weight", torch.zeros(self.d_model, self.d_model))
            self.register_buffer("depth_mlp_bias", torch.zeros(self.d_model))
            self.register_buffer("depth_fc_weight", torch.zeros(self.codebook_size, self.d_model))
            self.register_buffer("depth_fc_bias", torch.zeros(self.codebook_size))

        layers = []
        for lyr in prior._encoder.layers:
            nhead = int(lyr.self_attn.num_heads)
            layers.append(_KVLayer(lyr, self.d_model, nhead, self.max_len))
        self.layers = nn.ModuleList(layers)

        self.register_buffer("_len", torch.zeros((), dtype=torch.long))

    @torch.jit.export
    def reset(self):
        self._len.fill_(0)

    @torch.jit.export
    def cache_len(self) -> int:
        return int(self._len.item())

    def _context(self, tokens: torch.Tensor, territory: int = 0,
                 cond: Optional[torch.Tensor] = None,
                 territory_vec: Optional[torch.Tensor] = None,
                 style_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """tokens: [1, S, N] long. Returns the relu'd time-context [1, S, D]. Appends to cache.
        territory: index into the conditioning table (ignored when disabled).
        cond: optional [1, S, cond_dim] control envelope (the LFO) for these S positions.
        territory_vec: optional [1, d_model] continuous (interpolated) territory vector;
                       overrides the territory index when given.
        style_vec: optional [1, style_dim] global style code (from the XY pad); injected additively
                   at the input and as per-layer FiLM (mirrors PriorDiscrete._time_context)."""
        if tokens.dtype != torch.long:
            tokens = tokens.long()
        b, s, n = tokens.shape
        off = int(self._len.item())

        # Embedding + per-codebook offsets, [S, 1, N, E]
        x_sbn = tokens.permute(1, 0, 2)  # [S,1,N]
        offsets = self.codebook_offsets.view(1, 1, -1)
        idx = (x_sbn + offsets).clamp(min=0, max=self.vocab_per_codebook * self.num_codebooks - 1)
        emb = F.embedding(idx, self.embedding_weight)  # [S,1,N,E]

        # Absolute positional encoding pe[off:off+s], broadcast over codebooks.
        pe_slice = self.pe[off:off + s, :].view(s, 1, 1, self.embedding_dim)
        emb = emb + pe_slice
        x = emb.reshape(s, b, self.d_model)  # [S,1,D]

        # Territory conditioning: a provided (interpolated) vector wins; else the index.
        if territory_vec is not None:
            x = x + territory_vec.view(1, 1, self.d_model)
        elif self.has_territory:
            t = territory
            if t < 0:
                t = 0
            if t >= self.num_territories:
                t = self.num_territories - 1
            x = x + self.territory_weight[t].view(1, 1, self.d_model)

        # Envelope (LFO) conditioning: project per-position cond and add. cond: [1, S, cond_dim].
        if self.has_cond and cond is not None:
            cproj = F.linear(cond.float(), self.cond_proj_weight, self.cond_proj_bias)  # [1,S,D]
            x = x + cproj.permute(1, 0, 2)  # -> [S,1,D]

        # Style conditioning (global): add the projected style at the input (every position).
        if self.has_style and style_vec is not None:
            sp = F.linear(style_vec.view(1, self.style_dim), self.style_proj_weight, self.style_proj_bias)  # [1,D]
            x = x + sp.view(1, 1, self.d_model)

        # Additive causal mask [S, off+S]: query i (abs off+i) sees key j iff j <= off+i.
        total = off + s
        key_pos = torch.arange(total, device=x.device).view(1, total)
        qry_pos = (off + torch.arange(s, device=x.device)).view(s, 1)
        allowed = key_pos <= qry_pos
        attn_mask = torch.zeros(s, total, device=x.device)
        attn_mask = attn_mask.masked_fill(~allowed, float("-inf"))

        # Per-layer style FiLM (applied AFTER each layer, mirroring _time_context).
        li = 0
        for layer in self.layers:
            x = layer.decode(x, off, attn_mask)
            if self.has_style and style_vec is not None:
                gb = F.linear(style_vec.view(1, self.style_dim),
                              self.style_film_weight[li], self.style_film_bias[li])  # [1, 2D]
                g = gb[:, :self.d_model].view(1, 1, self.d_model)
                beta = gb[:, self.d_model:].view(1, 1, self.d_model)
                x = x * (1.0 + g) + beta
            li = li + 1

        x = x * self.sqrt_d
        x = x.permute(1, 0, 2)  # [1,S,D]

        x = F.relu(x)
        self._len.fill_(off + s)
        return x  # [1, S, D] time-context

    @torch.jit.export
    def decode(self, tokens: torch.Tensor, territory: int = 0,
               cond: Optional[torch.Tensor] = None,
               territory_vec: Optional[torch.Tensor] = None,
               style_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Independent-head logits [1, S, N, K] (non-joint models)."""
        h = self._context(tokens, territory, cond, territory_vec, style_vec)
        b = h.shape[0]; s = h.shape[1]
        fc = F.linear(h, self.fc_weight, self.fc_bias)
        return fc.view(b, s, self.num_codebooks, self.codebook_size)

    @torch.jit.export
    def decode_context(self, tokens: torch.Tensor, territory: int = 0,
                       cond: Optional[torch.Tensor] = None,
                       territory_vec: Optional[torch.Tensor] = None,
                       style_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Append tokens to the cache and return the relu'd time-context [1, S, D]
        (for joint-head sampling via depth_sample_last on the last position)."""
        return self._context(tokens, territory, cond, territory_vec, style_vec)

    @torch.jit.export
    def depth_sample_last(self, h_last: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        """Sample one frame's N codebooks autoregressively from time-context h_last [1, D].
        Returns tokens [1, N] long. Used for joint (WS4) models."""
        n = self.num_codebooks
        ksz = self.codebook_size
        prev = torch.zeros(1, self.d_model, device=h_last.device, dtype=h_last.dtype)
        toks = torch.zeros(1, n, dtype=torch.long, device=h_last.device)
        temp = temperature if temperature > 1e-4 else 1e-4
        for i in range(n):
            dpos = self.depth_pos_weight[i].view(1, self.d_model)
            hid = F.relu(F.linear(h_last + dpos + prev, self.depth_mlp_weight, self.depth_mlp_bias))
            li = F.linear(hid, self.depth_fc_weight, self.depth_fc_bias)  # [1, K]
            probs = F.softmax(li / temp, dim=-1)
            if top_p < 1.0:
                sp, si = torch.sort(probs, dim=-1, descending=True)
                csum = sp.cumsum(dim=-1)
                keep = (csum - sp) <= top_p
                sp = sp * keep
                sp = sp / sp.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                probs = torch.zeros_like(probs).scatter_(-1, si, sp)
            tok = torch.multinomial(probs, 1).view(1)  # [1]
            toks[:, i] = tok
            eidx = (tok + i * ksz).clamp(0, n * ksz - 1)
            prev = prev + F.embedding(eidx, self.depth_token_embed_weight)
        return toks

    @torch.jit.export
    def depth_sample_last_cfg(self, h_cond: torch.Tensor, h_uncond: torch.Tensor,
                              temperature: float, top_p: float, cfg_scale: float) -> torch.Tensor:
        """Classifier-free-guided joint sampling: per codebook,
        logits = uncond + cfg_scale*(cond - uncond). The SAME sampled token feeds both depth
        chains (so their running state stays identical). Returns tokens [1, N] long."""
        n = self.num_codebooks
        ksz = self.codebook_size
        prev = torch.zeros(1, self.d_model, device=h_cond.device, dtype=h_cond.dtype)
        toks = torch.zeros(1, n, dtype=torch.long, device=h_cond.device)
        temp = temperature if temperature > 1e-4 else 1e-4
        for i in range(n):
            dpos = self.depth_pos_weight[i].view(1, self.d_model)
            lc = F.linear(F.relu(F.linear(h_cond + dpos + prev, self.depth_mlp_weight, self.depth_mlp_bias)),
                          self.depth_fc_weight, self.depth_fc_bias)
            lu = F.linear(F.relu(F.linear(h_uncond + dpos + prev, self.depth_mlp_weight, self.depth_mlp_bias)),
                          self.depth_fc_weight, self.depth_fc_bias)
            li = lu + cfg_scale * (lc - lu)
            probs = F.softmax(li / temp, dim=-1)
            if top_p < 1.0:
                sp, si = torch.sort(probs, dim=-1, descending=True)
                csum = sp.cumsum(dim=-1)
                keep = (csum - sp) <= top_p
                sp = sp * keep
                sp = sp / sp.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                probs = torch.zeros_like(probs).scatter_(-1, si, sp)
            tok = torch.multinomial(probs, 1).view(1)
            toks[:, i] = tok
            eidx = (tok + i * ksz).clamp(0, n * ksz - 1)
            prev = prev + F.embedding(eidx, self.depth_token_embed_weight)
        return toks
