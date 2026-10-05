import logging

logger = logging.getLogger(__name__)

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch import Tensor
from timm.models.layers import trunc_normal_

from salma.model.modeling_finetune import PatchEmbed, Block, get_sinusoid_encoding_table


class ViTEncoderForSTAL(nn.Module):
    """
    Clean standalone Vision Transformer encoder for Spatio-Temporal Action Localization.

    This class essentialy replicates the encoder backbone from VideoMAE pretraining but:
        - Excludes the pretraining head and masking logic
        - Allows direct loading from a pretraining checkpoint
        - Provides a simplified forward pass (no mask input)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        init_values: Optional[float] = 0,
        cube_depth: int = 2,
        num_frames: int = 16,
        use_learnable_pos_emb: bool = False,
    ) -> None:
        
        super().__init__()

        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames = num_frames,
            tubelet_size=cube_depth,
        )

        # Positional embeddings (either fixed or learnable)
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = get_sinusoid_encoding_table(self.patch_embed.num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed.type_as(x).to(x.device)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def load_pretrained(self, ckpt_path: str) -> Tuple[List[str], List[str]]:
        """
        Load encoder weights from a VideoMAE pretraining checkpoint. This facilitates
        the use of iimporting weights from a pretraining run. 

        Args:
            ckpt_path (str): Path to checkpoint file ('.pth').

        Returns:
            Tuple[List[str], List[str]]: Missing and unexpected keys.
        """
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only = False)
        state_dict = checkpoint.get("model", checkpoint)

        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                if not "pos_embed" in k:
                    encoder_state_dict[k.replace("encoder.", "", 1)] = v
                else:
                    logger.info(f"Skipping positional embedding {k}")

        missing, unexpected = self.load_state_dict(encoder_state_dict, strict=False)
        logger.info(f"[load_pretrained] Missing: {len(missing)} | Unexpected: {len(unexpected)}")
        if missing:
            logger.info("  Missing keys: %s", missing)
        if unexpected:
            logger.info("  Unexpected keys: %s", unexpected)
        return missing, unexpected

    def get_num_layers(self):
        return len(self.blocks)
    