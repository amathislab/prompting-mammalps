import logging

logger = logging.getLogger(__name__)


import torch
import torch.nn as nn
from timm.models import register_model

from .ViTEncoderForSTAL import ViTEncoderForSTAL
from .TransformerDecoder import TransformerDecoder, TransformerDecoderLayer
from peft import LoraConfig, get_peft_model

from salma.model.modeling_finetune import get_sinusoid_encoding_table, Mlp
from salma.utils.misc import Mode


__all__ = [
    "salma_base_patch16_224",
    "salma_large_patch16_224",
]

class Salma(nn.Module):
    """
    Full spatio-temporal model for multi-head mammal understanding.

    This merges:
        - A ViT-like encoder producing cube (spatio-temporal patch) tokens
        - A DETR-style decoder with object queries propagated through time
        - Four prediction heads for boxes, species, actions, and activities
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder_embed_dim: int = 768,
        decoder_in_dim: int = 512,
        num_queries: int = 10,
        patch_grid: tuple[int, int] = (14, 14),
        tubelet_size: int = 2,
        num_species: int = 5,
        num_activities: int = 11,
        num_actions: int = 19,
        num_dage: int = 2,
        num_dsex: int = 2,
        num_weather: int = 4,
        mode: Mode = Mode.TRAIN
        ) -> None:

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.patch_grid = patch_grid
        self.tubelet_size = tubelet_size
        self.decoder_in_dim = decoder_in_dim
        self.num_queries = num_queries
        self.set_mode(mode=mode)

        # optional projection if encoder and decoder dims differ
        self.proj = (
            nn.Linear(encoder_embed_dim, decoder_in_dim)
            if encoder_embed_dim != decoder_in_dim
            else nn.Identity()
        )

        # learned object queries for first frame.
        self.first_animal_queries = nn.Embedding(num_queries, decoder_in_dim)

        # prediction heads at frame level
        self.spe_embed = Mlp(in_features = decoder_in_dim, hidden_features = decoder_in_dim // 4, out_features = num_species)
        self.actY_embed = Mlp(in_features = decoder_in_dim, hidden_features = decoder_in_dim // 4, out_features = num_activities)
        self.actN_embed = Mlp(in_features = decoder_in_dim, hidden_features = decoder_in_dim // 4, out_features = num_actions)
        self.DAge_embed = Mlp(in_features = decoder_in_dim, hidden_features = decoder_in_dim // 4, out_features = num_dage)
        self.DSex_embed = Mlp(in_features = decoder_in_dim, hidden_features = decoder_in_dim // 4, out_features = num_dsex)
        self.bbox_embed = Mlp(in_features = decoder_in_dim, hidden_features = decoder_in_dim // 4, out_features = 5)

        # prediction heads at video level
        self.weather_embed = Mlp(in_features=decoder_in_dim, hidden_features = decoder_in_dim // 4, out_features = num_weather)

        self.slot_animal_queries: dict[int, torch.Tensor] = {}
        self.slot_video_ids: dict[int, int] = {}   

    # positional embedding builder
    def build_pos(self, 
                  H_patches: int, 
                  W_patches: int, 
                  num_cubes: int, 
                  device: torch.device | str | None = None,
                  ) -> torch.Tensor:
        
        total = num_cubes * H_patches * W_patches
        return get_sinusoid_encoding_table(total, self.decoder_in_dim).to(device)
    
    def set_mode(self, mode: Mode):
        self.mode = mode
        
    def forward(
        self,
        images: torch.Tensor,
        slot_ids: list[int] = None,
        video_ids: list[int] = None,
    ) -> dict[str, torch.Tensor]:

        device = images.device
        B, C, N_frames, H, W = images.shape
        
        assert N_frames % self.tubelet_size == 0, "N_frames must be divisible by tubelet_size"

        num_cubes = N_frames // self.tubelet_size
        h, w = self.patch_grid


        # Encoder
        tokens = self.encoder(images)                # (B, num_cubes*h*w, D_enc)

        # Projecting to decoder input dim.
        tokens = self.proj(tokens)                   # (B, num_cubes*h*w, D_dec)
        pos_all = self.build_pos(h, w, num_cubes, tokens.device).expand(B, -1, -1)
        
        tokens_per_cube = tokens.view(B, num_cubes, h * w, self.decoder_in_dim)
        pos_per_cube = pos_all.view(B, num_cubes, h * w, self.decoder_in_dim)

        # weather predictions
        avg_tokens = tokens_per_cube.nanmean(dim=[1, 2], keepdims=True) 
        preds_weather = self.weather_embed(avg_tokens)


        # Handling animal queries updates.
        reset_q_single = self.first_animal_queries.weight.unsqueeze(1).to(device=device)  # (Q,1,D)
        if self.mode == Mode.TEST:
            active_queries = []

            for (slot_id, video_id) in zip(slot_ids, video_ids):
                reset = (
                    slot_id not in self.slot_video_ids
                    or self.slot_video_ids[slot_id] != video_id
                )

                if reset:
                    logger.debug(
                        f"Resetting queries for slot {slot_id} (new video {video_id})"
                    )
                    self.slot_animal_queries[slot_id] = reset_q_single.clone()
                    self.slot_video_ids[slot_id] = video_id

                active_queries.append(self.slot_animal_queries[slot_id])

            self.active_animal_queries = torch.cat(active_queries, dim=1)

        else:
            # reset all to learned queries (new object for safety)
            self.active_animal_queries = reset_q_single.repeat(1, B, 1).clone()    

        per_frame_preds = []
        per_frame_hs = []

        # Decoder
        for t in range(num_cubes):

            # We decode per cube to update the object queries
            mem = tokens_per_cube[:, t, :, :]
            pos = pos_per_cube[:, t, :, :]

            memory_t = mem.permute(1, 0, 2).contiguous()
            pos_t = pos.permute(1, 0, 2).contiguous()

            # Edited from DETR that builds a full 0 target and adds queries as pos embedding.
            hs = self.decoder(tgt=self.active_animal_queries, memory=memory_t, pos=pos_t, query_pos=None)

            # TransformerDecoder might return intermediate layers 
            final_hs = hs[-1] if hs.ndim == 4 else hs  # (Q, B, D)
            final_hs_B = final_hs.permute(1, 0, 2)     # (B, Q, D)

            # predictions
            pred_object_boxes = self.bbox_embed(final_hs_B)
            pred_boxes = pred_object_boxes[:,:,:4].sigmoid()
            pred_animals = pred_object_boxes[:,:,4]
            pred_species = self.spe_embed(final_hs_B)
            pred_activities = self.actY_embed(final_hs_B)
            pred_actions = self.actN_embed(final_hs_B)
            pred_dage = self.DAge_embed(final_hs_B)
            pred_dsex = self.DSex_embed(final_hs_B)

            self.active_animal_queries = final_hs_B.permute(1, 0, 2).contiguous().detach()

            per_frame_preds.append({
                "boxes": pred_boxes,
                "is_animal": pred_animals,
                "species": pred_species,
                "activities": pred_activities,
                "actions": pred_actions,
                "dage": pred_dage,
                "dsex": pred_dsex,
                "animal_queries": final_hs_B.contiguous()
            })
            per_frame_hs.append(final_hs)

        # Propagate queries
        if self.mode == Mode.TEST:
            for i, slot_id in enumerate(slot_ids):
                self.slot_animal_queries[slot_id] = self.active_animal_queries[:, i : i + 1, :].clone()

        # --- stack outputs ---
        all_boxes = torch.stack([p["boxes"] for p in per_frame_preds], dim=1)
        all_animals = torch.stack([p["is_animal"] for p in per_frame_preds], dim=1)
        all_species = torch.stack([p["species"] for p in per_frame_preds], dim=1)
        all_activities = torch.stack([p["activities"] for p in per_frame_preds], dim=1)
        all_actions = torch.stack([p["actions"] for p in per_frame_preds], dim=1)
        all_dage = torch.stack([p["dage"] for p in per_frame_preds], dim=1)
        all_dsex = torch.stack([p["dsex"] for p in per_frame_preds], dim=1)
        all_animal_queries = torch.stack([p["animal_queries"] for p in per_frame_preds], dim = 1)
        per_frame_hs = torch.stack(per_frame_hs, dim=0).permute(2, 0, 1, 3)
        return {
            "pred_boxes": all_boxes,
            "pred_is_animal": all_animals,
            "pred_species": all_species,
            "pred_activities": all_activities,
            "pred_actions": all_actions,
            "pred_dages": all_dage,
            "pred_dsexes": all_dsex,
            "pred_weather": preds_weather,
            "animal_queries": all_animal_queries,
            "per_frame_hs": per_frame_hs,
        }
    
def use_lora_encoder(model: nn.Module,
                     r: int = 16, 
                     lora_alpha: int = 16, 
                     dropout: float = 0.1):
    """
    Apply LoRA adapters to the encoder of a Salma-like model and log parameter statistics.
    """

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=[
            "attn.qkv",
            "attn.proj",
            "mlp.fc1",
            "mlp.fc2",
        ],
    )

    for name, param in model.encoder.named_parameters():
        param.requires_grad = False

    model.encoder = get_peft_model(model.encoder, config)
    
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
            print(name, type(module))

    logger.info("\n[LoRA] Encoder parameters summary:")
    model.encoder.print_trainable_parameters()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logger.info(f"\n[Model] Total parameters:     {total_params:,}")
    logger.info(f"[Model] Trainable parameters: {trainable_params:,}")
    logger.info(f"[Model] Frozen parameters:    {frozen_params:,}")
    logger.info(f"[Model] Trainable ratio:      {100 * trainable_params / total_params:.2f}%\n")

    return model        

    

@register_model
def salma_base_patch16_224(num_species: int,
                            num_activities: int,
                            num_actions: int,  
                            num_dage: int,
                            num_dsex: int,
                            pretrained: bool = False,
                            use_lora: bool = True,
                            lora_params: dict[str, float] = {}, 
                            num_queries: int = 10,
                            **kwargs
                            ) -> nn.Module:
    """
    Build a Salma with a ViT encoder and DETR-style decoder.
    """
    #TODO: Add dynamic config file to the model.
    encoder = ViTEncoderForSTAL(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        cube_depth=2,
        use_learnable_pos_emb=False,
    )

    if pretrained and "encoder_weights" in kwargs:
        encoder.load_pretrained(kwargs["encoder_weights"])

    decoder_layer = TransformerDecoderLayer(d_model = 512, 
                                             nhead = 8, 
                                             dim_feedforward = 1024,
                                             dropout = 0.1, 
                                             activation = "relu", 
                                             normalize_before = False)
    
    decoder_norm = nn.LayerNorm(512)

    decoder = TransformerDecoder(decoder_layer = decoder_layer, 
                                 num_layers = 4, 
                                 norm = decoder_norm,
                                 return_intermediate = False)

    model = Salma(
        encoder=encoder,
        decoder=decoder,
        encoder_embed_dim=768,
        decoder_in_dim=512,
        num_queries=num_queries,
        patch_grid=(14, 14),
        tubelet_size=2,
        num_species=num_species,
        num_activities=num_activities,
        num_actions=num_actions,
        num_dage=num_dage,
        num_dsex=num_dsex
    )

    if use_lora:
        logger.info("Using Low Rank Adaptation (LoRA) on encoder.")
        r = int(lora_params.get("r", 16))
        lora_alpha = int(lora_params.get("lora_alpha", 8))
        lora_dropout = lora_params.get("dropout", 0.1)

        model = use_lora_encoder(model = model, r = r, lora_alpha= lora_alpha, dropout = lora_dropout)
    else:
        logger.info("Not using Low Rank Adaptation (LoRA) on encoder.")
        
    return model


@register_model
def salma_large_patch16_224(num_species: int,
                            num_activities: int,
                            num_actions: int,  
                            num_dage: int,
                            num_dsex: int,
                            pretrained: bool = False,
                            use_lora: bool = True,
                            lora_params: dict[str, float] = {}, 
                            num_queries: int = 10,
                            **kwargs
                            ) -> nn.Module:
    """
    Build a Salma with a ViT encoder and DETR-style decoder.
    """
    #TODO: Add dynamic config file to the model.
    encoder = ViTEncoderForSTAL(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        cube_depth=2,
        use_learnable_pos_emb=False,
    )

    if pretrained and "encoder_weights" in kwargs:
        encoder.load_pretrained(kwargs["encoder_weights"])

    decoder_layer = TransformerDecoderLayer(d_model = 768, 
                                             nhead = 8, 
                                             dim_feedforward = 1024,
                                             dropout = 0.1, 
                                             activation = "relu", 
                                             normalize_before = False)
    
    decoder_norm = nn.LayerNorm(768)

    decoder = TransformerDecoder(decoder_layer = decoder_layer, 
                                 num_layers = 4, 
                                 norm = decoder_norm,
                                 return_intermediate = False)

    model = Salma(
        encoder=encoder,
        decoder=decoder,
        encoder_embed_dim=1024,
        decoder_in_dim=768,
        num_queries=num_queries,
        patch_grid=(14, 14),
        tubelet_size=2,
        num_species=num_species,
        num_activities=num_activities,
        num_actions=num_actions,
        num_dage=num_dage,
        num_dsex=num_dsex
    )

    if use_lora:
        logger.info("Using Low Rank Adaptation (LoRA) on encoder.")
        r = int(lora_params.get("r", 16))
        lora_alpha = int(lora_params.get("lora_alpha", 8))
        lora_dropout = lora_params.get("dropout", 0.1)

        model = use_lora_encoder(model = model, r = r, lora_alpha= lora_alpha, dropout = lora_dropout)
    else:
        logger.info("Not using Low Rank Adaptation (LoRA) on encoder.")
        
    return model