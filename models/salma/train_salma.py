import logging
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from timm import create_model
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

# Project-specific imports
from salma.losses.HungarianMatcher import HungarianMatcher
from salma.losses.SetCriterion import SetCriterion
from salma.datasets.stal_mammalps import (
    build_stal_mammalps,
    MammalpsSTALDistributedBalancingSampler,
    stal_collate_fn,
    )

from salma.model.engine_for_stal import train_one_epoch_stal, validation_one_epoch_stal
import salma.model.salma.Salma  # Needed to register model for timm.create_model
from salma.utils.misc import Mode, VideoFormat, load_label_mapping
from salma.model import utils
from salma.model.utils import init_distributed_mode




# -----------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train SALMA on MammAlps-S2")

    # --- Experiment setup ---
    p.add_argument("--experiment_name", type=str, default="default_exp")
    p.add_argument("--output_dir", type=str, default="./output")

    # --- Data paths ---
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--label_mapping_path", type=str, required=True)
    p.add_argument("--dense_annot_path", type=str, required=True)
    p.add_argument("--video_root_path", type=str, required=True)
    p.add_argument("--encoder_path", type=str, required=False)
    p.add_argument("--ckpt_path", type=str, required=False)
    p.add_argument("--log_dir", type=str, required=False)
    p.add_argument("--out_ckpt_prefix", type=str, default="ckpt_epoch_")

    # --- Model & training ---
    p.add_argument("--model_name", type=str, default="salma_base_patch16_224")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--val_freq", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--img_size", type=int, nargs=2, default=(224, 224))
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--num_queries", type=int, default=10)
    p.add_argument("--sampling_freq", type=int, default=2)
    p.add_argument("--vload_threads", type=int, default=1)
    p.add_argument("--min_rts", type=int, default=1)
    p.add_argument("--max_rts", type=int, default=1)
    p.add_argument("--no_val", action="store_true", help="No validation set")
    p.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder weights")
    p.add_argument("--freeze_decoder", action="store_true", help="Freeze decoder weights")
    p.add_argument("--freeze_object_heads", action="store_true", help="Freeze action and species heads")

    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup_lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int, default=40)

    p.add_argument("--weight_decay", type=float, default=0.001)
    p.add_argument("--weight_decay_end", type=float, default=None)

    p.add_argument("--save_ckpt_freq", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--print_freq", type=int, default=10)

    p.add_argument("--loss_species", type=float, default=1.)
    p.add_argument("--loss_activities", type=float, default=1.)
    p.add_argument("--loss_actions", type=float, default=1.)
    p.add_argument("--loss_dage", type=float, default=1.)
    p.add_argument("--loss_dsex", type=float, default=1.)
    p.add_argument("--loss_weather", type=float, default=1.)
    p.add_argument("--loss_l2_norm", type=float, default=0.01)
    p.add_argument("--loss_object", type=float, default=1.0)


    p.add_argument("--loss_bbox", type=float, default=5.)
    p.add_argument("--loss_giou", type=float, default=2.)
    p.add_argument("--loss_cont", type=float, default=0.3)
    p.add_argument("--eos_coef", type=float, default=0.3)

    # Sampler parameters
    p.add_argument("--balancing_sampling", action="store_true", default=False,
                        help='Enable weighted sampling based on inverse class frequency.')
    p.add_argument("--balancing_sampling_smoothing", type=float, default=1e-3,
                        help='Smoothing factor for balancing sampling weights.')

    p.add_argument("--use_lora", action="store_true", default=False,
                        help='Enable LoRA.')
    p.add_argument("--jitter", action="store_true", default=False,
                        help='Turns on temporal jittering during training.')
    p.add_argument("--rand_freq", action="store_true", default=False,
                        help='Turns on random sampling frequency during during training.')
    


    return p.parse_args()

def print_config(args, logger):
    # Log only from main process
    is_main_process = True
    if args.distributed and hasattr(args, "rank"):
            is_main_process = args.rank == 0

    if not is_main_process:
        return

    effective_batch_size = args.batch_size * getattr(args, "world_size", 1)

    logger.info("=" * 80)
    logger.info("Experiment Configuration")
    logger.info("=" * 80)

    # --- Experiment setup ---
    logger.info("[Experiment]")
    logger.info(f"  Experiment name        : {args.experiment_name}")
    logger.info(f"  Output directory       : {args.output_dir}")
    logger.info(f"  Seed                   : {args.seed}")
    logger.info(f"  Device                 : {args.device}")
    logger.info(f"  AMP enabled            : {args.amp}")
    logger.info(f"  World size             : {getattr(args, 'world_size', 1)}")
    logger.info("")

    # --- Data paths ---
    logger.info("[Data Paths]")
    logger.info(f"  CSV path               : {args.csv_path}")
    logger.info(f"  Label mapping path     : {args.label_mapping_path}")
    logger.info(f"  Dense annot path       : {args.dense_annot_path}")
    logger.info(f"  Video root path        : {args.video_root_path}")
    logger.info(f"  Encoder path           : {args.encoder_path}")
    logger.info(f"  Checkpoint path        : {args.ckpt_path}")
    logger.info(f"  Log directory          : {args.log_dir}")
    logger.info("")

    # --- Model & training ---
    logger.info("[Model & Training]")
    logger.info(f"  Model name             : {args.model_name}")
    logger.info(f"  Epochs                 : {args.epochs}")
    logger.info(f"  Batch size (per GPU)   : {args.batch_size}")
    logger.info(f"  Effective batch size   : {effective_batch_size}")
    logger.info(f"  Num workers            : {args.num_workers}")
    logger.info(f"  Image size             : {args.img_size}")
    logger.info(f"  Num frames             : {args.num_frames}")
    logger.info(f"  Num queries            : {args.num_queries}")
    logger.info(f"  Sampling frequency     : {args.sampling_freq}")
    logger.info(f"  Video load threads     : {args.vload_threads}")
    logger.info(f"  Skip validation        : {args.no_val}")
    logger.info(f"  Freeze encoder         : {args.freeze_encoder}")
    logger.info(f"  Freeze decoder         : {args.freeze_decoder}")
    logger.info(f"  Freeze object heads    : {args.freeze_object_heads}")
    
    logger.info(f"  Validation frequency   : {args.val_freq}")
    logger.info(f"  Print frequency        : {args.print_freq}")
    logger.info("")

    # --- Optimizer & LR schedule ---
    logger.info("[Optimization]")
    logger.info(f"  Learning rate          : {args.lr}")
    logger.info(f"  Min learning rate      : {args.min_lr}")
    logger.info(f"  Warmup learning rate   : {args.warmup_lr}")
    logger.info(f"  Warmup epochs          : {args.warmup_epochs}")
    logger.info(f"  Weight decay           : {args.weight_decay}")
    logger.info(f"  Weight decay end       : {args.weight_decay_end}")
    logger.info(f"  Save ckpt frequency    : {args.save_ckpt_freq}")
    logger.info("")

    # --- Loss weights ---
    logger.info("[Loss Weights]")
    logger.info(f"  Species loss           : {args.loss_species}")
    logger.info(f"  Activities loss        : {args.loss_activities}")
    logger.info(f"  Action loss            : {args.loss_actions}")
    logger.info(f"  DearAge loss           : {args.loss_dage}")
    logger.info(f"  AdultDeerSex loss      : {args.loss_dsex}")
    logger.info(f"  BBox loss              : {args.loss_bbox}")
    logger.info(f"  GIoU loss              : {args.loss_giou}")
    logger.info(f"  Contrastive loss       : {args.loss_cont}")
    logger.info(f"  L2 loss on object q.   : {args.loss_l2_norm}")
    logger.info(f"  EOS coefficient        : {args.eos_coef}")
    logger.info("")

    # --- Sampling & augmentation ---
    logger.info("[Sampling & Augmentation]")
    logger.info(f"  Balancing sampling     : {args.balancing_sampling}")
    logger.info(f"  Balancing smoothing    : {args.balancing_sampling_smoothing}")
    logger.info(f"  Temporal jitter        : {args.jitter}")
    logger.info(f"  Random freq sampling   : {args.rand_freq}")
    if args.rand_freq:
        logger.info(f"  Min RSF                : {args.min_rts}")
        logger.info(f"  Max RSF                : {args.max_rts}")
        
    logger.info("")

    # --- Model adaptations ---
    logger.info("[Model Adaptations]")
    logger.info(f"  Use LoRA               : {args.use_lora}")
    logger.info("")

    logger.info("=" * 80)

# -----------------------------------------------------------
# Model and criterion
# -----------------------------------------------------------
def make_model_and_criterion(args: argparse.Namespace, 
                            nb_species: int, 
                            nb_actions: int,
                            nb_activities: int,
                            nb_dage: int,
                            nb_dsex: int,
                            nb_weather: int,
                            device: str,
                            losses_list: list[str] = ["species", "activities", "actions", "dages", "dsexes", "boxes", "slot_contrast", "weather", "is_animal"],
                            hung_cost_class: float = 1.0,
                            hung_cost_bbox: float = 1.0,
                            hung_cost_giou: float = 1.0,
                            ) -> tuple[nn.Module, SetCriterion]:

    logger.info(f"Creating model {args.model_name} with pretrained encoder weights: {args.encoder_path}")
    logger.info(f"Loading previous checkpoint with checkpoint weights: {args.ckpt_path}")
    model = create_model(
        args.model_name,
        pretrained=True,
        encoder_weights=args.encoder_path,
        num_classes=None,
        use_lora=args.use_lora,
        num_queries=args.num_queries,
        num_species=nb_species,
        num_activities=nb_activities,
        num_actions=nb_actions, 
        num_dage=nb_dage,
        num_dsex=nb_dsex,
        num_weather=nb_weather
    )
    if args.ckpt_path:
        ckpt_state_dict = torch.load(args.ckpt_path, weights_only=False)
        model.load_state_dict(ckpt_state_dict["model_state_dict"])

    matcher = HungarianMatcher(cost_class=hung_cost_class, cost_bbox=hung_cost_bbox, cost_giou=hung_cost_giou)
    weight_dict = { "loss_species": args.loss_species, 
                    "loss_activities": args.loss_activities, 
                    "loss_actions": args.loss_actions, 
                    "loss_dage": args.loss_dage,
                    "loss_dsex": args.loss_dsex,
                    "loss_weather": args.loss_weather,
                    "loss_bbox": args.loss_bbox, 
                    "loss_giou": args.loss_giou, 
                    "loss_cont": args.loss_cont,
                    "loss_l2_norm": args.loss_l2_norm,
                    "loss_object": args.loss_object}

    logger.info(f"Using loss coefs: {weight_dict}")
    logger.info(f"Coefficient for none class: {args.eos_coef}")

    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses_list,
        device=device
    )

    model.to(device)
    return (model, criterion)


# -----------------------------------------------------------
# Data loader
# -----------------------------------------------------------
def build_dataloaders(args: argparse.Namespace, 
                      label_mapping: dict[str, dict[str, int]],
                      vload_threads: int = 1,
                      enable_jitter: bool = True,
                      min_rts: int = 1,
                      max_rts: int = 1,
                      ) -> tuple[DataLoader, Optional[DataLoader]]:

    dataset_train = build_stal_mammalps(
        csv_path=args.csv_path,
        dense_annot_root=args.dense_annot_path,
        label_mapping=label_mapping,
        video_root=args.video_root_path,
        mode=Mode.TRAIN,
        img_size=args.img_size,
        num_frames=args.num_frames,
        video_ext=VideoFormat.MP4,
        threads=vload_threads,
        min_stride=min_rts,
        max_stride=max_rts,
        jitter=enable_jitter
    )
    
    if args.balancing_sampling:
        sampler_train = MammalpsSTALDistributedBalancingSampler(
            dataset=dataset_train, 
            replacement=True,
            smoothing_factor=args.balancing_sampling_smoothing,
            seed=args.seed,
            rank=args.rank,
            drop_last=True
        )

    else:
        sampler_train = None

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=sampler_train,       
        collate_fn=stal_collate_fn,   
        drop_last=True,              
        num_workers=args.num_workers,
        pin_memory=True,
    )
   
    if not args.no_val:
        dataset_val = build_stal_mammalps(
            csv_path=args.csv_path,
            dense_annot_root=args.dense_annot_path,
            label_mapping=label_mapping,
            video_root=args.video_root_path,
            mode=Mode.VAL,
            img_size=args.img_size,
            num_frames=args.num_frames,        
            video_ext=VideoFormat.MP4,
            threads=vload_threads,
        )

        sampler_val = DistributedSampler(
            dataset_val,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
            drop_last=True,
        )


        loader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            collate_fn=stal_collate_fn,   
            sampler=sampler_val,
            drop_last=True,              
            num_workers=args.num_workers,
            pin_memory=True,
        )
        return (loader_train, loader_val)
    
    else:
        return (loader_train, None)


# -----------------------------------------------------------
# Main training loop
# -----------------------------------------------------------
def main():
    args = parse_args()
    init_distributed_mode(args)
    print_config(args, logger)


    device = torch.device(f"cuda:{args.gpu}")
    logger.debug(f"Running on device: {device}")

    rank = args.rank
    world_size = args.world_size

    out_dir_path = Path(args.output_dir)
    out_dir_path.mkdir(exist_ok= True, parents = True)

    log_path = out_dir_path / "log.txt"

    # Seed and device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Label mapping ---
    label_mapping = load_label_mapping(args.label_mapping_path, logger=logger)
    nb_species = len(label_mapping["species"])
    nb_actions = len(label_mapping["actions"])
    nb_activities = len(label_mapping["activities"])
    nb_dage = len(label_mapping["deer_ages"])
    nb_dsex = len(label_mapping["deer_adult_sexes"])
    nb_weather = len(label_mapping["weather"])

    # --- Data ---
    loader_train, loader_val = build_dataloaders(args=args, 
                                                                label_mapping=label_mapping,
                                                                vload_threads = args.vload_threads,
                                                                enable_jitter = args.jitter,
                                                                min_rts = args.min_rts,
                                                                max_rts = args.max_rts,
                                                                )


    num_training_steps_per_epoch = len(loader_train)

    logger.info(f"{rank}: Steps per epoch: {num_training_steps_per_epoch}")

    # --- Model & optimizer ---
    model, criterion = make_model_and_criterion(args=args, nb_species=nb_species, nb_actions=nb_actions, nb_activities=nb_activities, nb_dage=nb_dage, nb_dsex=nb_dsex, nb_weather=nb_weather, device=device)
    if args.distributed:
        model = DDP(model, 
                    device_ids=[args.gpu], 
                    output_device=args.gpu,
                    find_unused_parameters=True, 
                    )
        model_without_ddp = model.module

    else: 
        model_without_ddp = model

    heads_map = {"pred_species": "pred_species", 
                "pred_activities": "pred_activities", 
                "pred_actions": "pred_actions",
                "pred_dages": "pred_dages",
                "pred_dsexes": "pred_dsexes",
                "pred_boxes": "pred_boxes", 
                "pred_weather": "pred_weather",
                "pred_is_animal": "pred_is_animal",
                "animal_queries": "animal_queries"}
    
    log_writer = None
    if args.log_dir is not None:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)
        log_writer = utils.TensorboardLogger(log_dir=str(log_dir))

    # Scale LR with default batch size of 256
    total_batch_size = args.batch_size * world_size
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256

    # Set trainable parameters (Always freeze encoder for warmup epochs)
    for param in model_without_ddp.encoder.parameters():
        param.requires_grad = False
    if args.freeze_decoder:
        for param in model_without_ddp.decoder.parameters():
            param.requires_grad = False
    if args.freeze_object_heads:
        for param in model_without_ddp.spe_embed.parameters():
            param.requires_grad = False
        for param in model_without_ddp.actY_embed.parameters():
            param.requires_grad = False
        for param in model_without_ddp.actN_embed.parameters():
            param.requires_grad = False
        for param in model_without_ddp.DAge_embed.parameters():
            param.requires_grad = False
        for param in model_without_ddp.DSex_embed.parameters():
            param.requires_grad = False
        for param in model_without_ddp.weather_embed.parameters():
            param.requires_grad = False

    # Cosine scheduler for learning rate and weight decay
    logger.info("Using step level LR scheduler.")
    lr_schedule_values = utils.cosine_scheduler(
        base_value=args.lr,
        final_value=args.min_lr,
        epochs=args.epochs,
        niter_per_ep=num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        start_warmup_value=args.warmup_lr,
        warmup_steps=-1, # Computed in cosine scheduler.
    )
    logger.info("Max LR = %.7f, Min LR = %.7f" %
          (max(lr_schedule_values), min(lr_schedule_values)))
    
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay

    wd_schedule_values = utils.cosine_scheduler(base_value=args.weight_decay,
                                                final_value=args.weight_decay_end,
                                                epochs=args.epochs,
                                                niter_per_ep=num_training_steps_per_epoch
                                                )
    logger.info("Max WD = %.7f, Min WD = %.7f" %
          (max(wd_schedule_values), min(wd_schedule_values)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda') if args.amp and device.type == "cuda" else None
    
    if args.amp and device.type == "cuda":
        logger.info(f"Using GradScaler: {scaler}")

    # --- Training ---
    for epoch in range(args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        start = time.time()

        if epoch == args.warmup_epochs and not args.freeze_encoder:
            for param in model_without_ddp.encoder.parameters():
                param.requires_grad = True
        if isinstance(loader_train.sampler, MammalpsSTALDistributedBalancingSampler):
            loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_stal(
            model=model,
            criterion=criterion,
            heads_map=heads_map,
            data_loader=loader_train,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            loss_scaler=scaler,
            max_norm=1.0,
            model_ema=None,
            log_writer=log_writer,
            start_steps=epoch * len(loader_train),
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            print_freq=args.print_freq
        )

        elapsed = time.time() - start
        logger.info(f"[Epoch {epoch+1}/{args.epochs}] time={elapsed:.1f}s stats={train_stats}")

        # --- Validation (based on frequency) ---
        if loader_val is not None and (((epoch + 1) % args.val_freq == 0) or ((epoch + 1) == args.epochs)):
            val_stats = validation_one_epoch_stal(
                model = model,
                criterion = criterion,
                heads_map = heads_map,
                data_loader = loader_val,
                device = device,
                epoch = epoch,
                log_writer=log_writer,
                print_freq=args.print_freq,
            )
            logger.info(f"Validation results @ epoch {epoch+1}: {val_stats}")
        else:
            val_stats = {}

        if log_writer is not None:
                log_writer.flush()

        # Save checkpoint
        if (epoch + 1) % args.save_ckpt_freq == 0 or (epoch + 1) == args.epochs:
            ckpt_path = Path(args.output_dir) / f"{args.out_ckpt_prefix}_{epoch+1}.pth"
            if not (args.distributed and args.rank != 0):
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model_without_ddp.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                logger.info(f"Saved checkpoint: {ckpt_path}")

        # Log summary
        log_entry = {
            "epoch": epoch,
            "time_s": elapsed,
            **train_stats,
            "n_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    logger.info(f"Training completed. Logs saved to {log_path}")

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()