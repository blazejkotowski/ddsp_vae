from __future__ import annotations

import math
from typing import Any, Dict, Optional

import lightning as L
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import cross_entropy

from ddsp.prior.prior import FixedPositionalEncoding


class StyleEncoder(nn.Module):
    """Encode a window's token sequence [B,S,N] to a global style vector [B, style_dim].

    Mean-pools over time (no positional info) so it carries global *style* (texture/rhythm/grain),
    not the exact content. Bottlenecked + tanh-bounded. Trained jointly with the prior.
    """

    def __init__(self, num_codebooks: int, codebook_size: int, style_dim: int, hidden: int = 128):
        super().__init__()
        self._n = int(num_codebooks); self._k = int(codebook_size)
        self._embed = nn.Embedding(self._n * self._k, hidden)
        self.register_buffer("_off", torch.arange(self._n, dtype=torch.long) * self._k, persistent=False)
        self._net = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.ReLU(),
        )
        self._head = nn.Linear(hidden, int(style_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        idx = (tokens.long().clamp(0, self._k - 1) + self._off.view(1, 1, -1))  # [B,S,N]
        e = self._embed(idx).sum(2)                # [B,S,hidden]
        h = self._net(e.transpose(1, 2)).mean(-1)  # [B,hidden] global pool
        return torch.tanh(self._head(h))           # [B, style_dim]


class PriorDiscrete(L.LightningModule):
    """Causal Transformer prior over discrete VQ token sequences.

    Tokens are shaped [B, S, N] where N=num_codebooks and each token is in [0..codebook_size-1].
    """

    def __init__(
        self,
        *,
        num_codebooks: int,
        codebook_size: int,
        embedding_dim: int = 32,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 256,
        lr: float = 1e-4,
        num_territories: int = 0,
        cond_dim: int = 0,
        cfg_dropout: float = 0.0,
        cond_dropout: float = 0.0,
        joint_codebooks: bool = False,
        style_dim: int = 0,
        style_dropout: float = 0.0,
        context_dropout: float = 0.0,
        style_aux_weight: float = 0.0,
        ss_prob: float = 0.0,
        ss_anneal_steps: int = 0,
        ss_temperature: float = 1.0,
        ss_iters: int = 1,
        device: str = "cuda",
        **_ignored,  # absorb stale hyperparameters from older checkpoints (removed experiments)
    ):
        super().__init__()
        self.save_hyperparameters(ignore=list(_ignored.keys()))

        self._num_codebooks = int(num_codebooks)
        self._codebook_size = int(codebook_size)
        self._embedding_dim = int(embedding_dim)
        self._num_layers = int(num_layers)
        self._d_model = int(self._embedding_dim * self._num_codebooks)
        self._lr = float(lr)
        self._max_len = int(max_len)
        self._num_territories = int(num_territories)
        # Classifier-free guidance: train with territory-dropout to a learned NULL/uncond row
        # (index == num_territories) so the model learns both p(x) and p(x|territory). At
        # inference, guide logits = uncond + scale*(cond - uncond) to amplify track-faithfulness.
        self._cfg_dropout = float(cfg_dropout)
        self._cfg_null = int(self._num_territories)  # null/uncond territory index
        # LFO cond-dropout: zero the cond envelope for a fraction of training windows so the model
        # learns BOTH LFO-driven and freeform (cond=0) generation -> switchable / blendable at inference.
        self._cond_dropout = float(cond_dropout)
        # Territory conditioning: an additive embedding per position (+1 row for the CFG null territory).
        self._territory_embedding = (
            nn.Embedding(self._num_territories + 1, self._d_model)
            if self._num_territories > 0 else None
        )
        # Time-varying conditioning on a slow control envelope (the LFO/scaffold at inference):
        # project the per-position cond vector to d_model and add it to every position.
        self._cond_dim = int(cond_dim)
        self._cond_proj = nn.Linear(self._cond_dim, self._d_model) if self._cond_dim > 0 else None

        # Learned global STYLE code: a bottlenecked summary of a window's character, encoded from its
        # tokens, injected at every position (+ FiLM) and trained jointly. With the explicit LFO/cond
        # present, the LFO "explains away" the slow envelope so the style learns the residual texture/
        # rhythm. style_dropout (>0) zeros it sometimes -> a clean null for style-CFG at inference.
        self._style_dim = int(style_dim)
        self._style_dropout = float(style_dropout)
        # context_dropout: corrupt a fraction of the context tokens during training so local context
        # alone can't predict the next token -> the prior must lean on the global style code.
        self._context_dropout = float(context_dropout)
        # Scheduled sampling: feed the prior its OWN (iteratively re-sampled) tokens in the context so it
        # learns to recover from its own distribution at free-running. ss_iters>1 re-samples the context
        # over several passes so it carries realistic multi-step drift. ss_prob ramps 0 -> target.
        self._ss_prob = float(ss_prob)
        self._ss_anneal_steps = int(ss_anneal_steps)
        self._ss_temperature = float(ss_temperature)
        self._ss_iters = max(1, int(ss_iters))
        if self._style_dim > 0:
            self._style_encoder = StyleEncoder(self._num_codebooks, self._codebook_size, self._style_dim)
            self._style_proj = nn.Linear(self._style_dim, self._d_model)          # additive at the input
            # Per-layer FiLM: modulate the hidden (scale+shift) AFTER every transformer layer so the
            # style pathway is strong and persistent (load-bearing at free-running generation).
            self._style_film = nn.ModuleList(
                [nn.Linear(self._style_dim, 2 * self._d_model) for _ in range(self._num_layers)])
            for lin in self._style_film:
                nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)  # start as identity modulation
            # Style-discriminability auxiliary: classify the source track from the style code so the
            # encoder makes `s` maximally track-separable. Only meaningful with per-track territories.
            self._style_aux_weight = float(style_aux_weight)
            self._style_clf = (nn.Linear(self._style_dim, self._num_territories)
                               if (self._style_aux_weight > 0.0 and self._num_territories > 0) else None)
        else:
            self._style_encoder = None; self._style_proj = None; self._style_film = None
            self._style_aux_weight = 0.0; self._style_clf = None

        # A learned START token per codebook (id == codebook_size) gives generation
        # an in-distribution cold start instead of a zero/random primer.
        self._start_id = self._codebook_size
        self._vocab_per_codebook = self._codebook_size + 1  # +1 for START

        # One shared embedding table with per-codebook offsets.
        vocab_size = self._vocab_per_codebook * self._num_codebooks
        self._embedding = nn.Embedding(vocab_size, self._embedding_dim)
        self.register_buffer(
            "_codebook_offsets",
            torch.arange(self._num_codebooks, dtype=torch.long) * self._vocab_per_codebook,
            persistent=False,
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=self._d_model,
            nhead=nhead,
            dim_feedforward=int(dim_feedforward),
            dropout=dropout,
        )
        self._encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._positional_encoding = FixedPositionalEncoding(
            embedding_dim=self._embedding_dim,
            max_len=self._max_len,
            dropout=dropout,
            device=device,
        )
        self.register_buffer(
            "_causal_mask_full",
            torch.triu(torch.ones(self._max_len, self._max_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

        self._activation = nn.ReLU()
        self._fc = nn.Linear(self._d_model, self._num_codebooks * self._codebook_size)

        # WS4 joint codebook head: predict codebook i conditioned on the already-decoded
        # codebooks <i WITHIN the same timestep, so sampled frames stay on-manifold (kills the
        # off-manifold clicks from independent per-codebook sampling). The time-context h from the
        # transformer feeds a small "depth" decoder over the N codebooks: parallel via an exclusive
        # prefix-sum of decoded-codebook embeddings in training, sequential (N steps) at inference.
        self._joint = bool(joint_codebooks)
        if self._joint:
            self._depth_pos = nn.Embedding(self._num_codebooks, self._d_model)
            self._depth_token_embed = nn.Embedding(self._num_codebooks * self._codebook_size, self._d_model)
            self.register_buffer(
                "_depth_offsets",
                torch.arange(self._num_codebooks, dtype=torch.long) * self._codebook_size,
                persistent=False,
            )
            self._depth_mlp = nn.Linear(self._d_model, self._d_model)
            self._depth_fc = nn.Linear(self._d_model, self._codebook_size)
            self._depth_act = nn.ReLU()

    @property
    def num_codebooks(self) -> int:
        return self._num_codebooks

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def start_token_id(self) -> int:
        return self._start_id

    def territory_embedding_table(self) -> Optional[torch.Tensor]:
        """[num_territories+1, d_model] table, for building interpolated territory vectors."""
        return None if self._territory_embedding is None else self._territory_embedding.weight

    def _encode_style(self, tokens: torch.Tensor) -> torch.Tensor:
        """Global style vector [B, style_dim] from a window's tokens."""
        return self._style_encoder(tokens)

    def _time_context(self, x_tokens: torch.Tensor, territory_id: Optional[torch.Tensor] = None,
                      cond: Optional[torch.Tensor] = None,
                      territory_vec: Optional[torch.Tensor] = None,
                      style_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Transformer time-context h: [B, S, D] (post conditioning and activation).

        This is the per-timestep hidden the output head(s) read from; the independent and the
        joint (depth) heads both consume it.
        """
        if x_tokens.dtype != torch.long:
            x_tokens = x_tokens.long()

        b, s, n = x_tokens.shape
        if n != self._num_codebooks:
            raise ValueError(f"Expected num_codebooks={self._num_codebooks}, got {n}")
        if s > self._max_len:
            x_tokens = x_tokens[:, -self._max_len :, :]
            s = x_tokens.shape[1]

        # [S, B, N]
        x = x_tokens.permute(1, 0, 2)

        offsets = self._codebook_offsets.view(1, 1, -1)
        idx = (x + offsets).clamp(min=0, max=self._vocab_per_codebook * self._num_codebooks - 1)

        # [S, B, N, E]
        embed = self._embedding(idx)
        embed = self._positional_encoding(embed)

        # [S, B, D]
        pos = embed.reshape(s, b, self._d_model)

        # Territory conditioning: a provided (possibly interpolated) vector wins over the index.
        if territory_vec is not None:
            pos = pos + territory_vec.view(b, self._d_model).unsqueeze(0)
        elif self._territory_embedding is not None and territory_id is not None:
            terr = self._territory_embedding(territory_id.long().view(b))  # [B, D]
            pos = pos + terr.unsqueeze(0)

        # Envelope conditioning: add the projected per-position control envelope.
        if self._cond_proj is not None and cond is not None:
            # cond: [B, S, cond_dim] -> [S, B, D]
            cproj = self._cond_proj(cond[:, -s:, :].float()).permute(1, 0, 2)
            pos = pos + cproj

        # Style conditioning (global): add the projected style to every position at the input.
        sv = None
        if self._style_proj is not None and style_vec is not None:
            sv = style_vec.view(b, self._style_dim)
            pos = pos + self._style_proj(sv).unsqueeze(0)

        causal_mask = self._causal_mask_full[:s, :s]
        # Run the transformer layer-by-layer so style FiLM is applied AFTER each layer (persistent grip).
        for i, layer in enumerate(self._encoder.layers):
            pos = layer(pos, src_mask=causal_mask)  # [S, B, D]
            if self._style_film is not None and sv is not None:
                g, beta = self._style_film[i](sv).chunk(2, dim=-1)  # [B, D] each
                pos = pos * (1.0 + g.unsqueeze(0)) + beta.unsqueeze(0)
        if self._encoder.norm is not None:
            pos = self._encoder.norm(pos)
        enc = pos * math.sqrt(self._d_model)

        # [B, S, D]
        enc = enc.permute(1, 0, 2)
        enc = self._activation(enc)
        return enc  # [B, S, D]

    def forward(self, x_tokens: torch.Tensor, territory_id: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None,
                territory_vec: Optional[torch.Tensor] = None,
                depth_targets: Optional[torch.Tensor] = None,
                style_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return logits [B, S, N, K] for next-token prediction.

        For a joint model, pass `depth_targets` [B, S, N] (the true tokens at each position) for
        teacher-forced depth logits (training). At inference the joint head is sampled incrementally
        via `depth_decode_last` instead of calling forward.
        """
        h = self._time_context(x_tokens, territory_id, cond, territory_vec, style_vec)
        b, s = h.shape[0], h.shape[1]
        if self._joint:
            return self._depth_logits_tf(h, depth_targets[:, -s:, :])
        fc = self._fc(h)
        return fc.view(b, s, self._num_codebooks, self._codebook_size)

    def _depth_embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed already-decoded codebook tokens with per-codebook offsets. tokens [..., N] long."""
        n = self._num_codebooks
        idx = (tokens.long() + self._depth_offsets.view(*([1] * (tokens.dim() - 1)), n))
        idx = idx.clamp(min=0, max=n * self._codebook_size - 1)
        return self._depth_token_embed(idx)  # [..., N, D]

    def _depth_logits_tf(self, h: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Teacher-forced joint logits, parallel over codebooks. h [B,S,D], targets [B,S,N]."""
        b, s, d = h.shape
        n = self._num_codebooks
        e = self._depth_embed(targets)                  # [B,S,N,D]
        pre = torch.cumsum(e, dim=2) - e                # exclusive prefix sum over codebooks
        dpos = self._depth_pos(torch.arange(n, device=h.device)).view(1, 1, n, d)
        c = h.unsqueeze(2) + dpos + pre                 # [B,S,N,D]
        hid = self._depth_act(self._depth_mlp(c))
        return self._depth_fc(hid)                      # [B,S,N,K]

    def _depth_sample_all(self, h: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample the N codebooks for EVERY position in parallel (joint head, N sequential steps).

        h [B,S,D] -> sampled tokens [B,S,N]. Used for scheduled sampling during training (feed the
        model its own samples). The exclusive prefix-sum of decoded-codebook embeddings is built
        incrementally from the SAMPLED tokens (vs the true targets in _depth_logits_tf).
        """
        b, s, d = h.shape
        n = self._num_codebooks
        dpos = self._depth_pos(torch.arange(n, device=h.device)).view(1, 1, n, d)
        sampled = torch.zeros(b, s, n, dtype=torch.long, device=h.device)
        pre = torch.zeros(b, s, d, device=h.device)  # running exclusive prefix sum of decoded embeds
        temp = max(1e-4, float(temperature))
        for i in range(n):
            c = h + dpos[:, :, i, :] + pre                       # [B,S,D]
            logits = self._depth_fc(self._depth_act(self._depth_mlp(c)))  # [B,S,K]
            probs = torch.softmax(logits / temp, dim=-1)
            tok_i = torch.multinomial(probs.reshape(-1, self._codebook_size), 1).reshape(b, s)
            sampled[:, :, i] = tok_i
            e_i = self._depth_token_embed(tok_i + int(self._depth_offsets[i].item()))  # [B,S,D]
            pre = pre + e_i
        return sampled

    def depth_decode_last(self, h_last: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0,
                          h_last_uncond: Optional[torch.Tensor] = None, cfg_scale: float = 1.0) -> torch.Tensor:
        """Sample one frame's N codebooks autoregressively from the time-context h_last [B,D].

        If h_last_uncond is given with cfg_scale!=1, classifier-free guidance is applied per
        codebook (the same sampled token feeds both conditional and unconditional chains).
        Returns tokens [B, N] long.
        """
        b, d = h_last.shape
        n, ksz = self._num_codebooks, self._codebook_size
        prev = torch.zeros(b, d, device=h_last.device, dtype=h_last.dtype)
        toks = []
        use_cfg = h_last_uncond is not None and float(cfg_scale) != 1.0
        for i in range(n):
            dpos = self._depth_pos.weight[i].view(1, d)
            li = self._depth_fc(self._depth_act(self._depth_mlp(h_last + dpos + prev)))  # [B,K]
            if use_cfg:
                lu = self._depth_fc(self._depth_act(self._depth_mlp(h_last_uncond + dpos + prev)))
                li = lu + float(cfg_scale) * (li - lu)
            probs = torch.softmax(li / max(1e-4, float(temperature)), dim=-1)
            if top_p < 1.0:
                sp, si = torch.sort(probs, dim=-1, descending=True)
                csum = sp.cumsum(dim=-1)
                keep = csum - sp <= top_p
                sp = sp * keep
                sp = sp / sp.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                probs = torch.zeros_like(probs).scatter_(-1, si, sp)
            tok = torch.multinomial(probs, 1).squeeze(-1)  # [B]
            toks.append(tok)
            prev = prev + self._depth_token_embed((tok + i * ksz).clamp(0, n * ksz - 1))
        return torch.stack(toks, dim=1)  # [B, N]

    @property
    def is_joint(self) -> bool:
        return self._joint

    def time_context(self, x_tokens: torch.Tensor, territory_id: Optional[torch.Tensor] = None,
                     cond: Optional[torch.Tensor] = None,
                     territory_vec: Optional[torch.Tensor] = None,
                     style_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Public accessor for the [B,S,D] time-context (used by samplers)."""
        return self._time_context(x_tokens, territory_id, cond, territory_vec, style_vec)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        out = self._step(batch)
        self.log("loss", out["loss"], prog_bar=True)
        self.log("acc", out["acc"], prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return out

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        out = self._step(batch)
        self.log("val_loss", out["loss"], prog_bar=True)
        self.log("val_acc", out["acc"], prog_bar=True)
        return out

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=20,
            threshold=1e-4,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }

    def _step(self, batch) -> Dict[str, torch.Tensor]:
        # batch: [B, S, N], or (tokens, extra) where extra is territory_id [B] (long)
        # or a control envelope cond [B, S, cond_dim] (float).
        territory_id = None
        cond = None
        if isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                batch, cond, territory_id = batch[0], batch[1], batch[2]
            else:
                extra = batch[1]
                batch = batch[0]
                if torch.is_floating_point(extra):
                    cond = extra
                else:
                    territory_id = extra
        # CFG territory-dropout: randomly route some samples to the NULL territory so the model
        # learns the unconditioned distribution alongside the conditioned ones.
        if (territory_id is not None and self._cfg_dropout > 0.0 and self.training
                and self._territory_embedding is not None):
            territory_id = territory_id.long().clone()
            drop = torch.rand(territory_id.shape[0], device=territory_id.device) < self._cfg_dropout
            territory_id[drop] = self._cfg_null
        # LFO cond-dropout: zero the cond for some samples so cond=0 means in-distribution freeform.
        if cond is not None and self._cond_dropout > 0.0 and self.training:
            cond = cond.clone()
            cdrop = torch.rand(cond.shape[0], device=cond.device) < self._cond_dropout
            cond[cdrop] = 0.0
        if batch.dtype != torch.long:
            batch = batch.long()

        b, s, n = batch.shape
        # Prepend a START token so the model learns P(first token | START).
        start = torch.full((b, 1, n), self._start_id, dtype=torch.long, device=batch.device)
        seq = torch.cat([start, batch], dim=1)  # [B, S+1, N]
        x = seq[:, :-1, :]  # [B, S, N], begins with START
        y = seq[:, 1:, :]   # [B, S, N], the real tokens

        style_vec = None
        style_aux = None
        if self._style_encoder is not None:
            # Style is the GLOBAL texture of this window (the LFO-complement): encode it from the
            # window's CLEAN real tokens (before any context corruption). Explaining-away — the
            # explicit LFO/cond captures the slow envelope, so `s` carries residual texture/rhythm.
            s_clean = self._encode_style(batch)
            # Discriminability auxiliary on the CLEAN style (before dropout): classify the source track.
            # Exclude rows routed to the CFG-null territory (label == num_territories is out of range).
            if self._style_clf is not None and territory_id is not None:
                tlab = territory_id.long().view(b)
                valid = tlab < self._num_territories
                if bool(valid.any()):
                    style_aux = cross_entropy(self._style_clf(s_clean[valid]), tlab[valid])
            style_vec = s_clean
            if self.training and self._style_dropout > 0.0:
                sdrop = torch.rand(style_vec.shape[0], device=style_vec.device) < self._style_dropout
                style_vec = style_vec.clone()
                style_vec[sdrop] = 0.0

        # Context dropout: corrupt a fraction of context positions so local context alone can't
        # predict y -> the prior must lean on the (clean) global style code. START (pos 0) is kept.
        if self.training and self._context_dropout > 0.0:
            cmask = torch.rand(b, x.shape[1], device=x.device) < self._context_dropout
            cmask[:, 0] = False
            rand_tok = torch.randint(0, self._codebook_size, x.shape, dtype=x.dtype, device=x.device)
            x = torch.where(cmask.unsqueeze(-1), rand_tok, x)

        # Scheduled sampling: replace a fraction of context tokens with the model's OWN (iteratively
        # re-sampled) tokens so it learns to consume its own drifted distribution at free-running.
        if self._joint and self.training and self._ss_prob > 0.0:
            ss_p = self._ss_prob
            if self._ss_anneal_steps > 0:
                ss_p = ss_p * min(1.0, float(int(self.global_step)) / float(self._ss_anneal_steps))
            if ss_p > 0.0:
                # Iterated SS: each pass re-samples from the progressively-drifted context so the model
                # sees the realistic free-running DRIFT (loudness flattening) paired with the sharp targets.
                with torch.no_grad():
                    x_cur = x
                    for _ in range(self._ss_iters):
                        h_it = self._time_context(x_cur, territory_id, cond, None, style_vec)
                        samp = self._depth_sample_all(h_it, self._ss_temperature)  # [B,S,N] guesses for t_j
                        smask = torch.rand(b, x_cur.shape[1], device=x.device) < ss_p
                        smask[:, 0] = False  # keep START
                        # x position p (>=1) holds t_{p-1}; replace with the own guess samp[:, p-1]
                        x_next = x_cur.clone()
                        x_next[:, 1:, :] = torch.where(smask[:, 1:].unsqueeze(-1), samp[:, :-1, :], x_cur[:, 1:, :])
                        x_cur = x_next
                x = x_cur

        dt = y if self._joint else None
        logits = self(x, territory_id=territory_id, cond=cond, depth_targets=dt,
                      style_vec=style_vec)  # [B, S, N, K]
        y_hat = torch.argmax(logits, dim=-1)
        acc = (y_hat == y).float().mean()

        k = self._codebook_size
        ce = cross_entropy(
            logits.permute(0, 3, 1, 2).reshape(b, k, -1),
            y.reshape(b, -1),
            reduction="none",
        ).nanmean()

        loss = ce
        if style_aux is not None:
            loss = loss + self._style_aux_weight * style_aux
            return {"loss": loss, "acc": acc, "ce": ce.detach(), "style_aux": style_aux.detach()}
        return {"loss": loss, "acc": acc, "ce": ce.detach()}
