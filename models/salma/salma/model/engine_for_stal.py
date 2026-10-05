import logging

logger = logging.getLogger(__name__)

import math
import sys
from typing import Iterable

import torch
from timm.utils import ModelEma

import salma.model.utils as utils  


def target_to_device(targets: list[dict[str, torch.Tensor]], device: torch.device | str | None = None):
    """
    Move a list of DETR-style target dicts to device.
    targets: list of dicts with keys 'boxes' (tensor Nx4), 'labels' (tensor N), optional 'frames'
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    new_targets = []
    for t in targets:
        t_dev = {}
        for k, v in t.items():
            if isinstance(v, torch.Tensor):
                t_dev[k] = v.to(device, non_blocking=True)
            else:
                t_dev[k] = v
        new_targets.append(t_dev)
    return new_targets

def flatten_model_outputs_for_heads(outputs: dict[str, torch.Tensor], heads: dict[str, str]):
    """
    Convert model outputs to the format expected by SetCriterion.
    heads: mapping from desired criterion key -> model output key
           e.g. {'pred_logits': 'pred_species', 'pred_boxes': 'pred_boxes'}
    returns a dict with keys 'pred_logits', 'pred_boxes', etc flattened on batch/time dims.
    Assumes model outputs are shaped (B, T', Q, C) or (B, T', Q, 4).
    """

    out = {}
    for key, model_key in heads.items():
        x = outputs[model_key]
        if (key != "animal_queries"):
            # flatten batch and time dims -> (B*T', Q, ...)
            x = x.flatten(0, 1)
            
        out[key] = x
    return out

def train_one_epoch_stal(model: torch.nn.Module,
                         criterion: torch.nn.Module,
                         heads_map: dict[str,str],
                         data_loader: Iterable,
                         optimizer: torch.optim.Optimizer,
                         epoch: int = 0,
                         device: torch.device | str | None = None,
                         loss_scaler: torch.amp.GradScaler | None = None,
                         max_norm: float = 0.0,
                         model_ema: ModelEma | None = None,
                         log_writer: utils.TensorboardLogger = None,
                         start_steps: int = 0,
                         lr_schedule_values = None,
                         wd_schedule_values = None,
                         num_training_steps_per_epoch: int | None = None,
                         print_freq: int = 10
                         ) -> dict[str, float]:
    """
    Train loop for one epoch for the STAL DETR-based model.

    - heads_map maps the criterion keys to model output keys, e.g.
        {'pred_logits': 'pred_species', 'pred_boxes': 'pred_boxes'}
    - data_loader is expected to yield tuples like (data_dict,) or (samples, raw_targets, idxs)
      We support these shapes:
        - (data,) where data is the collated data dict used earlier (data["video"], data["boxes"], data["labels"])
        - (samples, raw_targets, idxs) -> similar to other code
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    model.train(True)
    model = model.to(device)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    weight_dict = criterion.weight_dict
    total_weight = sum(v for (k,v) in weight_dict.items())


    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):  
        step = data_iter_step

        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for param_group in optimizer.param_groups:
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] 
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        data = batch
        
        if isinstance(data, dict):
            samples = data["video"].to(device, non_blocking=True)
            raw_targets = data["targets"]
            weather_targets = [t[0]["weather"].to(device) for t in raw_targets]
            targets_flat = [item for sublist in raw_targets for item in sublist]
            targets = target_to_device(targets = targets_flat, device = device)

        else:
            raise RuntimeError("Unsupported dataloader batch type")

        # Forward pass
        with torch.autocast(device_type=device.type):
            outputs = model(samples)
            # Preparing outputs dict expected by SetCriterion
            outputs_for_criterion = flatten_model_outputs_for_heads(outputs, heads_map)
            loss_dict = criterion(outputs_for_criterion, targets, weather_targets)

            total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) / total_weight

        loss_value = total_loss.item()

        if not math.isfinite(loss_value):
            logger.error(loss_dict)
            logger.error(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        if loss_scaler is None:
            # Standard backward
            total_loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)
            loss_scale_value = 1.0
            grad_norm = utils.get_grad_norm(model.parameters()) if hasattr(utils, "get_grad_norm") else 0.0

        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

            scaled_loss = loss_scaler.scale(total_loss)
            scaled_loss.backward(create_graph=is_second_order)

            if max_norm > 0:
                loss_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            loss_scaler.step(optimizer)
            loss_scaler.update()
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

            loss_scale_value = loss_scaler.get_scale()
            grad_norm = utils.get_grad_norm(model.parameters()) if hasattr(utils, "get_grad_norm") else 0.0

        # logging basic losses
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        for k, v in loss_dict.items():
            metric_logger.update(**{k: v.item()})

        # learning rate logging
        min_lr = min(group["lr"] for group in optimizer.param_groups)
        max_lr = max(group["lr"] for group in optimizer.param_groups)
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        metric_logger.update(grad_norm=grad_norm)

    # end epoch
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # Epoch-wise logging (averaged)
    if log_writer is not None:
        avg_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        for k, v in avg_stats.items():
            log_writer.update(**{k: v}, head="opt")
        log_writer.set_step(epoch) 

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch_stal(model: torch.nn.Module,
                              criterion: torch.nn.Module,
                              heads_map: dict[str, str],
                              data_loader: Iterable,
                              device: torch.device | str | None = None,
                              epoch: int = 0,
                              log_writer: utils.TensorboardLogger = None,
                              print_freq: int = 10):
    """
    Validation loop for STAL DETR-based model.

    Matches the style of `train_one_epoch_stal`, logs metrics to TensorBoard,
    and returns averaged validation statistics.
    """
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Validation: [Epoch {epoch}]'

    weight_dict = criterion.weight_dict
    total_weight = sum(v for (k,v) in weight_dict.items())

    for batch in metric_logger.log_every(data_loader, print_freq, header):

        if isinstance(batch, dict):
            video = batch["video"]
            if isinstance(video, list):
                video = torch.stack(video)
            samples = video.to(device, non_blocking=True)
            raw_targets = batch["targets"]
            weather_targets = [t[0]["weather"].to(device) for t in raw_targets]
            targets_flat = [item for sublist in raw_targets for item in sublist]
            targets = target_to_device(targets = targets_flat, device = device)

        else:
            raise RuntimeError("Unsupported dataloader batch type (expected dict).")

        # Forward pass
        with torch.autocast(device_type=device.type):
            outputs = model(samples)

            # Preparing outputs dict expected by SetCriterion
            outputs_for_criterion = flatten_model_outputs_for_heads(outputs, heads_map)

            loss_dict = criterion(outputs_for_criterion, targets, weather_targets)

            total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) / total_weight

        loss_value = total_loss.item()
        # --- Metric logging ---
        loss_value = total_loss.item()
        metric_logger.update(loss=loss_value)
        for k, v in loss_dict.items():
            metric_logger.update(**{k: v.item()})

    metric_logger.synchronize_between_processes()
    print("Validation averaged stats:", metric_logger)

    if log_writer is not None:
        avg_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        for k, v in avg_stats.items():
            log_writer.update(**{k: v}, head="val")
        log_writer.set_step(epoch)  # one step per epoch

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

