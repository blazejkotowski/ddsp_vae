"""
Convolutional encoder-decoder for compressing DDSP control sequences.
689 Hz -> ~20 Hz (~34x compression) and back.
"""
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


class ResidualBlock(nn.Module):
    """1D residual block with two convolutions."""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class ConvEncoder(nn.Module):
    """Strided convolutional encoder for downsampling."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        compressed_dim: int,
        strides: List[int],
        num_residual_layers: int = 2,
        kernel_size: int = 7,
        max_channels: Optional[int] = None,
    ):
        super().__init__()
        self.strides = strides
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Downsampling stages
        self.down_blocks = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()
        
        ch = hidden_dim
        for stride in strides:
            out_ch = ch * 2
            if max_channels is not None:
                out_ch = min(int(out_ch), int(max_channels))

            # Strided conv for downsampling
            self.down_blocks.append(
                nn.Conv1d(ch, out_ch, kernel_size=kernel_size, stride=stride,
                         padding=(kernel_size - 1) // 2)
            )
            ch = out_ch

            # Residual blocks at this resolution
            res_layers = nn.ModuleList([
                ResidualBlock(ch) for _ in range(num_residual_layers)
            ])
            self.residual_blocks.append(res_layers)
        
        # Output projection to compressed_dim
        self.output_proj = nn.Conv1d(ch, compressed_dim, kernel_size=1)
        self.activation = nn.LeakyReLU(0.2)
        self.final_channels = ch
        self.final_channels = ch
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # x: [B, T, input_dim] -> [B, input_dim, T]
        x = x.transpose(1, 2)
        
        x = self.activation(self.input_proj(x))
        
        skip_connections = []
        for down, res_blocks in zip(self.down_blocks, self.residual_blocks):
            skip_connections.append(x)
            x = self.activation(down(x))
            for res in res_blocks:
                x = res(x)
        
        x = self.output_proj(x)
        # x: [B, compressed_dim, T_compressed]
        return x, skip_connections


class ConvDecoder(nn.Module):
    """Transposed convolutional decoder for upsampling."""
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        compressed_dim: int,
        strides: List[int],
        num_residual_layers: int = 2,
        kernel_size: int = 7,
        use_skip_connections: bool = True,
        start_channels: Optional[int] = None,
    ):
        super().__init__()
        self.strides = list(reversed(strides))
        self.use_skip_connections = use_skip_connections
        
        # Start from encoder final channels unless overridden.
        ch = int(start_channels) if start_channels is not None else (hidden_dim * (2 ** len(strides)))
        
        # Input projection from compressed_dim
        self.input_proj = nn.Conv1d(compressed_dim, ch, kernel_size=1)
        
        # Upsampling stages
        self.up_blocks = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()
        # Keep as ModuleList even when unused so TorchScript can type-check indexing.
        self.skip_projs = nn.ModuleList()
        
        for stride in self.strides:
            # Residual blocks at this resolution
            res_layers = nn.ModuleList([
                ResidualBlock(ch) for _ in range(num_residual_layers)
            ])
            self.residual_blocks.append(res_layers)
            
            # Skip connection projection (encoder has ch//2 at this level)
            if use_skip_connections:
                self.skip_projs.append(nn.Conv1d(ch // 2, ch, kernel_size=1))
            else:
                self.skip_projs.append(nn.Identity())
            
            # Transposed conv for upsampling
            self.up_blocks.append(
                nn.ConvTranspose1d(ch, ch // 2, kernel_size=kernel_size, stride=stride,
                                  padding=(kernel_size - 1) // 2, output_padding=stride - 1)
            )
            ch //= 2
        
        # Output projection
        self.output_proj = nn.Conv1d(ch, output_dim, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor, skip_connections: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        # x: [B, compressed_dim, T_compressed]
        x = self.activation(self.input_proj(x))

        # Reverse skip connections to match decoder order (TorchScript-friendly)
        if self.use_skip_connections and skip_connections is not None:
            rev = torch.jit.annotate(List[torch.Tensor], [])
            for j in range(len(skip_connections) - 1, -1, -1):
                rev.append(skip_connections[j])
            skip_connections = rev

        for i, (up, res_blocks, skip_proj) in enumerate(zip(self.up_blocks, self.residual_blocks, self.skip_projs)):
            for res in res_blocks:
                x = res(x)

            # Add skip connection before upsampling
            if self.use_skip_connections and skip_connections is not None:
                skip = skip_connections[i]
                # Project skip to match channels
                skip = skip_proj(skip)
                # Interpolate if sizes don't match
                if skip.shape[-1] != x.shape[-1]:
                    skip = F.interpolate(skip, size=x.shape[-1], mode='linear', align_corners=False)
                x = x + skip

            x = self.activation(up(x))
        
        x = self.output_proj(x)
        # x: [B, output_dim, T] -> [B, T, output_dim]
        return x.transpose(1, 2)


class VectorQuantizer(nn.Module):
    """Single-codebook VQ with EMA codebook updates + dead-code restart.

    Anti-collapse design (vs a plain straight-through VQ):
      * the codebook is updated by an exponential moving average of the encoder
        outputs assigned to each code, not by gradient (only a commitment loss
        flows to the encoder);
      * the codebook is initialised from the first batch of encoder outputs;
      * codes that go (almost) unused are periodically re-seeded from live encoder
        outputs.
    These keep all codes alive on low-diversity / small datasets, where a plain
    VQ collapses to a handful of codes.

    Interface is unchanged:
      forward(z_e:[B,C,T]) -> (z_q_st:[B,C,T], indices:[B,T], vq_loss, perplexity)
      embed(indices:[B,T]) -> z_q:[B,C,T]
    """

    def __init__(self, embedding_dim: int, codebook_size: int, beta: float = 0.25,
                 decay: float = 0.99, eps: float = 1e-5, restart_threshold: float = 1.0):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.codebook_size = int(codebook_size)
        self.beta = float(beta)
        self.decay = float(decay)
        self.eps = float(eps)
        self.restart_threshold = float(restart_threshold)

        embed = torch.randn(self.codebook_size, self.embedding_dim) * 0.1
        # Codebook + EMA stats live in buffers (updated in-place, not via autograd).
        self.register_buffer("codebook", embed)
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("_initted", torch.zeros((), dtype=torch.bool))

    def embed(self, indices: torch.Tensor) -> torch.Tensor:
        """Map code indices [B, T] -> quantized vectors [B, C, T]."""
        if indices.dtype != torch.long:
            indices = indices.long()
        z_q = F.embedding(indices, self.codebook)  # [B, T, C]
        return z_q.permute(0, 2, 1).contiguous()

    @torch.jit.unused
    def _data_init(self, z_flat: torch.Tensor) -> None:
        """Initialise the codebook from the first batch of encoder outputs (training only)."""
        with torch.no_grad():
            n = z_flat.shape[0]
            if n >= self.codebook_size:
                idx = torch.randperm(n, device=z_flat.device)[: self.codebook_size]
            else:
                idx = torch.randint(0, n, (self.codebook_size,), device=z_flat.device)
            chosen = z_flat[idx].detach()
            self.codebook.copy_(chosen)
            self.embed_avg.copy_(chosen)
            self.cluster_size.fill_(1.0)
            self._initted.fill_(True)

    @torch.jit.unused
    def _ema_update(self, z_flat: torch.Tensor, onehot: torch.Tensor) -> None:
        """EMA codebook update + dead-code restart (training only)."""
        with torch.no_grad():
            code_count = onehot.sum(0)              # [K]
            embed_sum = onehot.t() @ z_flat         # [K, C]
            self.cluster_size.mul_(self.decay).add_(code_count, alpha=1.0 - self.decay)
            self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)
            # Laplace-smoothed normalisation of the codebook.
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
            self.codebook.copy_(self.embed_avg / cluster_size.unsqueeze(1))
            # Dead-code restart: re-seed underused codes from live encoder outputs.
            dead = self.cluster_size < self.restart_threshold
            n_dead = int(dead.sum())
            if n_dead > 0:
                m = z_flat.shape[0]
                ridx = torch.randint(0, m, (n_dead,), device=z_flat.device)
                samples = z_flat[ridx].detach()
                self.codebook[dead] = samples
                self.embed_avg[dead] = samples
                self.cluster_size[dead] = 1.0

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e: [B, C, T] continuous latents
        Returns:
            z_q_st: [B, C, T] quantized latents (straight-through)
            indices: [B, T] code indices
            vq_loss: scalar (commitment only; codebook learned via EMA)
            perplexity: scalar
        """
        B, C, T = z_e.shape
        z = z_e.permute(0, 2, 1).contiguous()  # [B, T, C]
        z_flat = z.reshape(B * T, C)           # [N, C]

        if self.training and not bool(self._initted):
            self._data_init(z_flat)

        # Squared L2 distance to each codebook vector.
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.codebook.pow(2).sum(dim=1).unsqueeze(0)
            - 2.0 * (z_flat @ self.codebook.t())
        )
        indices = torch.argmin(dist, dim=1)  # [N]

        z_q_flat = F.embedding(indices, self.codebook)  # [N, C]
        z_q = z_q_flat.view(B, T, C).permute(0, 2, 1).contiguous()  # [B, C, T]

        onehot = F.one_hot(indices, num_classes=self.codebook_size).type(z_flat.dtype)  # [N, K]

        if self.training:
            self._ema_update(z_flat, onehot)

        # Only the commitment loss trains the encoder; the codebook is EMA-updated.
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = self.beta * commitment_loss

        # Straight-through estimator.
        z_q_st = z_e + (z_q - z_e).detach()

        # Perplexity (effective number of codes used in this batch).
        avg_probs = onehot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, indices.view(B, T), vq_loss, perplexity


class GroupedVectorQuantizer(nn.Module):
    """Product-quantization style: split channels into groups, one codebook per group.

    This implements the "4 indices per timestep" idea (e.g., num_codebooks=4).
    """

    def __init__(
        self,
        embedding_dim: int,
        codebook_size: int,
        num_codebooks: int,
        beta: float = 0.25,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.codebook_size = int(codebook_size)
        self.num_codebooks = int(num_codebooks)
        self.beta = float(beta)

        if self.num_codebooks < 1:
            raise ValueError(f"num_codebooks must be >= 1, got {self.num_codebooks}")
        if self.embedding_dim % self.num_codebooks != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by num_codebooks ({self.num_codebooks})"
            )

        self.group_dim = self.embedding_dim // self.num_codebooks
        self.quantizers = nn.ModuleList(
            [VectorQuantizer(self.group_dim, self.codebook_size, beta=self.beta) for _ in range(self.num_codebooks)]
        )

    def embed(self, indices: torch.Tensor) -> torch.Tensor:
        """Map grouped indices to quantized vectors.

        Args:
            indices: [B, T, N] long
        Returns:
            z_q: [B, C, T]
        """
        if indices.dtype != torch.long:
            indices = indices.long()
        if indices.dim() != 3 or indices.shape[-1] != self.num_codebooks:
            raise RuntimeError("Expected indices shape [B, T, num_codebooks]")

        z_q_groups = torch.jit.annotate(List[torch.Tensor], [])
        for i, q in enumerate(self.quantizers):
            idx = indices[..., i]  # [B, T]
            z_q_groups.append(q.embed(idx))
        return torch.cat(z_q_groups, dim=1)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize by splitting channels into groups.

        Args:
            z_e: [B, C, T]
        Returns:
            z_q: [B, C, T]
            indices: [B, T, N]
            vq_loss: scalar (sum over codebooks)
            perplexity: scalar (mean over codebooks)
        """
        z_qs = torch.jit.annotate(List[torch.Tensor], [])
        idxs = torch.jit.annotate(List[torch.Tensor], [])
        losses = torch.jit.annotate(List[torch.Tensor], [])
        perps = torch.jit.annotate(List[torch.Tensor], [])

        for i, q in enumerate(self.quantizers):
            start = i * self.group_dim
            end = start + self.group_dim
            z_g = z_e[:, start:end, :]
            z_q, idx, vq_loss, perp = q(z_g)
            z_qs.append(z_q)
            idxs.append(idx)
            losses.append(vq_loss)
            perps.append(perp)

        z_q = torch.cat(z_qs, dim=1)
        indices = torch.stack(idxs, dim=-1)  # [B, T, N]
        vq_loss = torch.stack(losses).mean()
        perplexity = torch.stack(perps).mean()
        return z_q, indices, vq_loss, perplexity


class LatentCompressor(L.LightningModule):
    """
    Convolutional encoder-decoder for compressing DDSP control sequences.
    
    Args:
        input_dim: int (n_features + n_latents, e.g., 4 for hybrid)
        hidden_dim: int (channel count in conv layers)
        compressed_dim: int (bottleneck dimension)
        strides: List[int] (e.g., [8, 4] for 32x compression)
        num_residual_layers: int
        learning_rate: float
        use_skip_connections: bool
    """
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        compressed_dim: int = 16,
        strides: List[int] = [8, 4],
        num_residual_layers: int = 2,
        kernel_size: int = 7,
        max_channels: Optional[int] = None,
        learning_rate: float = 1e-4,
        use_skip_connections: bool = True,
        vq_enabled: bool = False,
        vq_codebook_size: int = 1024,
        vq_beta: float = 0.25,
        vq_loss_weight: float = 1.0,
        vq_num_codebooks: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Ensure learning_rate is float (YAML might pass string)
        self.learning_rate = float(learning_rate)

        self.use_skip_connections = bool(use_skip_connections)
        self.vq_enabled = bool(vq_enabled)
        self.vq_loss_weight = float(vq_loss_weight)
        self.vq_num_codebooks = int(vq_num_codebooks)

        if self.vq_enabled and self.use_skip_connections:
            raise ValueError(
                "VQ bottleneck requires use_skip_connections=False for an AR-prior-compatible codes-only decoder."
            )

        self.encoder = ConvEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            compressed_dim=compressed_dim,
            strides=strides,
            num_residual_layers=num_residual_layers,
            kernel_size=kernel_size,
            max_channels=max_channels,
        )
        
        self.decoder = ConvDecoder(
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            compressed_dim=compressed_dim,
            strides=strides,
            num_residual_layers=num_residual_layers,
            kernel_size=kernel_size,
            use_skip_connections=self.use_skip_connections,
            start_channels=getattr(self.encoder, "final_channels", None),
        )

        self.vq: Optional[nn.Module] = None
        if self.vq_enabled:
            if self.vq_num_codebooks <= 1:
                self.vq = VectorQuantizer(
                    embedding_dim=compressed_dim,
                    codebook_size=vq_codebook_size,
                    beta=vq_beta,
                )
            else:
                self.vq = GroupedVectorQuantizer(
                    embedding_dim=compressed_dim,
                    codebook_size=vq_codebook_size,
                    num_codebooks=self.vq_num_codebooks,
                    beta=vq_beta,
                )
        
        self.compression_ratio = 1
        for s in strides:
            self.compression_ratio *= s
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: [B, T_high, input_dim] - control sequences at 689 Hz
        Returns:
            x_hat: [B, T_high, input_dim] - reconstructed controls
            aux: dict with keys: z (quantized), indices, vq_loss, perplexity
        """
        z_e, skips = self.encoder(x)

        if self.vq is not None:
            # Clamp encoder latents to a stable range before quantization.
            z_e = torch.tanh(z_e)
            z_q, indices, vq_loss, perplexity = self.vq(z_e)
        else:
            z_q = z_e
            indices = z_e.new_zeros((z_e.shape[0], z_e.shape[-1]), dtype=torch.long)
            vq_loss = z_e.new_tensor(0.0)
            perplexity = z_e.new_tensor(0.0)

        x_hat = self.decoder(z_q, skips if self.use_skip_connections else None)

        # Ensure output matches input length
        T_in = x.shape[1]
        T_out = x_hat.shape[1]
        if T_out > T_in:
            x_hat = x_hat[:, :T_in, :]
        elif T_out < T_in:
            x_hat = F.pad(x_hat, (0, 0, 0, T_in - T_out))

        aux = {
            'z': z_q,
            'indices': indices,
            'vq_loss': vq_loss,
            'perplexity': perplexity,
        }
        return x_hat, aux

    def encode_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Encode controls to discrete code indices.

        Returns:
            indices: [B, T_low] for single codebook, or [B, T_low, N] for grouped VQ.
        """
        if self.vq is None:
            raise RuntimeError("encode_codes requires vq_enabled=True")

        z_e, _ = self.encoder(x)
        z_e = torch.tanh(z_e)
        _, indices, _, _ = self.vq(z_e)
        return indices

    def decode_codes(self, indices: torch.Tensor, output_len: Optional[int] = None) -> torch.Tensor:
        """Decode discrete code indices back to controls.

        Args:
            indices: [B, T_low] or [B, T_low, N]
            output_len: optionally crop/pad to this many high-rate steps.
        Returns:
            x_hat: [B, T_high, input_dim]
        """
        if self.vq is None:
            raise RuntimeError("decode_codes requires vq_enabled=True")
        if self.use_skip_connections:
            raise RuntimeError("decode_codes requires use_skip_connections=False (codes-only decoding)")

        z_q = self.vq.embed(indices)
        x_hat = self.decoder(z_q, None)

        if output_len is not None:
            T_out = x_hat.shape[1]
            if T_out > output_len:
                x_hat = x_hat[:, :output_len, :]
            elif T_out < output_len:
                x_hat = F.pad(x_hat, (0, 0, 0, output_len - T_out))

        return x_hat
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_hat, aux = self(batch)
        recon_loss = F.mse_loss(x_hat, batch)
        loss = recon_loss + self.vq_loss_weight * aux['vq_loss']
        self.log('train_loss', loss, prog_bar=False)
        self.log('train_recon_loss', recon_loss, prog_bar=True)
        self.log('train_vq_loss', aux['vq_loss'], prog_bar=False)
        self.log('train_perplexity', aux['perplexity'], prog_bar=False)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_hat, aux = self(batch)
        recon_loss = F.mse_loss(x_hat, batch)
        self.log('val_loss', recon_loss, prog_bar=True)
        self.log('val_vq_loss', aux['vq_loss'], prog_bar=False)
        self.log('val_perplexity', aux['perplexity'], prog_bar=False)
        return recon_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }
    
    def get_num_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
